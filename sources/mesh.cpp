#include "mesh.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

// cuFFT header
#include <cufft.h>

// Constructor
Mesh::Mesh(size_t gridSize, double gridSpacing)
    : gridSize(gridSize), gridSpacing(gridSpacing),
      deviceDensity(nullptr), devicePotential(nullptr),
      deviceDensityFFT(nullptr), devicePotentialFFT(nullptr),
      forwardPlan(0), inversePlan(0)
{

    // Set grid origin (typically at 0,0,0 for simplicity)
    gridOrigin[0] = gridOrigin[1] = gridOrigin[2] = 0.0;

    // Initialize host arrays
    size_t totalPoints = getTotalGridPoints();
    hostDensity.resize(totalPoints, 0.0);
    hostPotential.resize(totalPoints, 0.0);
    hostForceX.resize(totalPoints, 0.0);
    hostForceY.resize(totalPoints, 0.0);
    hostForceZ.resize(totalPoints, 0.0);
}

// Destructor
Mesh::~Mesh()
{
    freeDeviceMemory(*(new CUDAManager())); // Dummy manager for cleanup
    destroyFFTPlans();
}

// Initialize the mesh
bool Mesh::initialize(CUDAManager &cudaMgr)
{
    if (!cudaMgr.isInitialized())
    {
        std::cerr << "CUDA manager not initialized" << std::endl;
        return false;
    }

    // Allocate device memory
    if (!allocateDeviceMemory(cudaMgr))
    {
        std::cerr << "Failed to allocate device memory for mesh" << std::endl;
        return false;
    }

    // Create FFT plans
    if (!createFFTPlans(cudaMgr))
    {
        std::cerr << "Failed to create FFT plans" << std::endl;
        return false;
    }

    return true;
}

// Allocate device memory
bool Mesh::allocateDeviceMemory(CUDAManager &cudaMgr)
{
    size_t totalPoints = getTotalGridPoints();
    size_t complexPoints = totalPoints / 2 + 1; // For R2C FFT

    bool success = true;
    success &= cudaMgr.allocateDeviceMemory(&deviceDensity, totalPoints);
    success &= cudaMgr.allocateDeviceMemory(&devicePotential, totalPoints);
    success &= cudaMgr.allocateDeviceMemory(&deviceDensityFFT, complexPoints * sizeof(cufftDoubleComplex) / sizeof(double));
    success &= cudaMgr.allocateDeviceMemory(&devicePotentialFFT, complexPoints * sizeof(cufftDoubleComplex) / sizeof(double));

    if (!success)
    {
        std::cerr << "Failed to allocate device memory for mesh" << std::endl;
        freeDeviceMemory(cudaMgr);
        return false;
    }

    return true;
}

// Free device memory
bool Mesh::freeDeviceMemory(CUDAManager &cudaMgr)
{
    bool success = true;

    if (deviceDensity)
    {
        success &= cudaMgr.freeDeviceMemory(deviceDensity);
        deviceDensity = nullptr;
    }

    if (devicePotential)
    {
        success &= cudaMgr.freeDeviceMemory(devicePotential);
        devicePotential = nullptr;
    }

    if (deviceDensityFFT)
    {
        success &= cudaMgr.freeDeviceMemory(deviceDensityFFT);
        deviceDensityFFT = nullptr;
    }

    if (devicePotentialFFT)
    {
        success &= cudaMgr.freeDeviceMemory(devicePotentialFFT);
        devicePotentialFFT = nullptr;
    }

    return success;
}

// Check if device memory is allocated
bool Mesh::isDeviceMemoryAllocated() const
{
    return deviceDensity && devicePotential && deviceDensityFFT && devicePotentialFFT;
}

// Create FFT plans
bool Mesh::createFFTPlans(CUDAManager &cudaMgr)
{
    // Destroy existing plans if any
    destroyFFTPlans();

    // Create 3D FFT plans
    cufftResult result;

    // Forward plan: real to complex
    result = cufftPlan3d(&forwardPlan, gridSize, gridSize, gridSize, CUFFT_D2Z);
    if (result != CUFFT_SUCCESS)
    {
        std::cerr << "Failed to create forward FFT plan: " << result << std::endl;
        return false;
    }

    // Inverse plan: complex to real
    result = cufftPlan3d(&inversePlan, gridSize, gridSize, gridSize, CUFFT_Z2D);
    if (result != CUFFT_SUCCESS)
    {
        std::cerr << "Failed to create inverse FFT plan: " << result << std::endl;
        destroyFFTPlans();
        return false;
    }

    return true;
}

// Destroy FFT plans
bool Mesh::destroyFFTPlans()
{
    bool success = true;

    if (forwardPlan)
    {
        cufftResult result = cufftDestroy(forwardPlan);
        if (result != CUFFT_SUCCESS)
        {
            std::cerr << "Warning: Failed to destroy forward FFT plan" << std::endl;
            success = false;
        }
        forwardPlan = 0;
    }

    if (inversePlan)
    {
        cufftResult result = cufftDestroy(inversePlan);
        if (result != CUFFT_SUCCESS)
        {
            std::cerr << "Warning: Failed to destroy inverse FFT plan" << std::endl;
            success = false;
        }
        inversePlan = 0;
    }

    return success;
}

// Assign particles to mesh using CIC interpolation
bool Mesh::assignParticlesToMesh(const std::vector<double> &positions,
                                 const std::vector<double> &masses,
                                 size_t numParticles)
{
    if (positions.size() != numParticles * 3 || masses.size() != numParticles)
    {
        std::cerr << "Invalid input data sizes" << std::endl;
        return false;
    }

    // Reset density field
    resetDensity();

    // Apply CIC assignment
    applyCICPAssignment(positions, masses, numParticles);

    return true;
}

// Apply CIC (Cloud-in-Cell) particle assignment
void Mesh::applyCICPAssignment(const std::vector<double> &positions,
                               const std::vector<double> &masses,
                               size_t numParticles)
{
    for (size_t p = 0; p < numParticles; ++p)
    {
        double px = positions[p * 3];
        double py = positions[p * 3 + 1];
        double pz = positions[p * 3 + 2];
        double mass = masses[p];

        // Convert to grid coordinates
        double gx = px / gridSpacing;
        double gy = py / gridSpacing;
        double gz = pz / gridSpacing;

        // Get integer part (grid cell indices)
        int i = static_cast<int>(floor(gx));
        int j = static_cast<int>(floor(gy));
        int k = static_cast<int>(floor(gz));

        // Get fractional part (interpolation weights)
        double dx = gx - i;
        double dy = gy - j;
        double dz = gz - k;

        // CIC weights for the 8 neighboring cells
        double wx[2] = {1.0 - dx, dx};
        double wy[2] = {1.0 - dy, dy};
        double wz[2] = {1.0 - dz, dz};

        // Distribute mass to 8 neighboring grid points
        for (int di = 0; di < 2; ++di)
        {
            for (int dj = 0; dj < 2; ++dj)
            {
                for (int dk = 0; dk < 2; ++dk)
                {
                    int gi = (i + di) % static_cast<int>(gridSize);
                    int gj = (j + dj) % static_cast<int>(gridSize);
                    int gk = (k + dk) % static_cast<int>(gridSize);

                    // Handle negative indices
                    if (gi < 0)
                        gi += gridSize;
                    if (gj < 0)
                        gj += gridSize;
                    if (gk < 0)
                        gk += gridSize;

                    double weight = wx[di] * wy[dj] * wz[dk];
                    size_t gridIndex = getGridIndex(gi, gj, gk);
                    hostDensity[gridIndex] += mass * weight;
                }
            }
        }
    }
}

// Solve Poisson equation using FFT
bool Mesh::solvePoissonFFT()
{
    if (!isDeviceMemoryAllocated() || !forwardPlan || !inversePlan)
    {
        std::cerr << "Mesh not properly initialized" << std::endl;
        return false;
    }

    // Copy density to device
    CUDAManager dummyMgr; // We need a proper CUDA manager here
    // For now, assume density is already on device
    // In a real implementation, we'd copy hostDensity to deviceDensity

    cufftResult result;

    // Forward FFT: density -> densityFFT
    result = cufftExecD2Z(forwardPlan, deviceDensity, deviceDensityFFT);
    if (result != CUFFT_SUCCESS)
    {
        std::cerr << "Forward FFT failed: " << result << std::endl;
        return false;
    }

    // Apply Green function in Fourier space
    // This would multiply densityFFT by the Green function
    // For now, we'll implement a simplified version

    // Inverse FFT: potentialFFT -> potential
    result = cufftExecZ2D(inversePlan, devicePotentialFFT, devicePotential);
    if (result != CUFFT_SUCCESS)
    {
        std::cerr << "Inverse FFT failed: " << result << std::endl;
        return false;
    }

    // Normalize by grid size^3
    double normalization = 1.0 / getTotalGridPoints();
    // Apply normalization on device (would need a kernel)

    return true;
}

// Calculate forces from potential gradients
bool Mesh::calculateForcesFromPotential()
{
    // This would compute -∇Φ on the grid
    // For now, implement a simplified version

    size_t totalPoints = getTotalGridPoints();

    for (size_t idx = 0; idx < totalPoints; ++idx)
    {
        size_t i, j, k;
        getGridCoordinates(idx, i, j, k);

        // Compute gradients using finite differences
        // This is a simplified implementation

        // Force in x direction: -dΦ/dx
        double phi_left = (i > 0) ? hostPotential[getGridIndex(i - 1, j, k)] : hostPotential[getGridIndex(gridSize - 1, j, k)];
        double phi_right = (i < gridSize - 1) ? hostPotential[getGridIndex(i + 1, j, k)] : hostPotential[getGridIndex(0, j, k)];
        hostForceX[idx] = (phi_left - phi_right) / (2.0 * gridSpacing);

        // Force in y direction: -dΦ/dy
        double phi_down = (j > 0) ? hostPotential[getGridIndex(i, j - 1, k)] : hostPotential[getGridIndex(i, gridSize - 1, k)];
        double phi_up = (j < gridSize - 1) ? hostPotential[getGridIndex(i, j + 1, k)] : hostPotential[getGridIndex(i, 0, k)];
        hostForceY[idx] = -(phi_up - phi_down) / (2.0 * gridSpacing);

        // Force in z direction: -dΦ/dz
        double phi_back = (k > 0) ? hostPotential[getGridIndex(i, j, k - 1)] : hostPotential[getGridIndex(i, j, gridSize - 1)];
        double phi_front = (k < gridSize - 1) ? hostPotential[getGridIndex(i, j, k + 1)] : hostPotential[getGridIndex(i, j, 0)];
        hostForceZ[idx] = -(phi_front - phi_back) / (2.0 * gridSpacing);
    }

    return true;
}

// Interpolate forces to particles using CIC
bool Mesh::interpolateForcesToParticles(const std::vector<double> &positions,
                                        std::vector<double> &forces,
                                        size_t numParticles)
{
    if (positions.size() != numParticles * 3 || forces.size() != numParticles * 3)
    {
        std::cerr << "Invalid input/output data sizes" << std::endl;
        return false;
    }

    // Apply CIC interpolation
    applyCICInterpolation(positions, forces, numParticles);

    return true;
}

// Apply CIC interpolation for forces
void Mesh::applyCICInterpolation(const std::vector<double> &positions,
                                 std::vector<double> &forces,
                                 size_t numParticles)
{
    for (size_t p = 0; p < numParticles; ++p)
    {
        double px = positions[p * 3];
        double py = positions[p * 3 + 1];
        double pz = positions[p * 3 + 2];

        // Convert to grid coordinates
        double gx = px / gridSpacing;
        double gy = py / gridSpacing;
        double gz = pz / gridSpacing;

        // Get integer part (grid cell indices)
        int i = static_cast<int>(floor(gx));
        int j = static_cast<int>(floor(gy));
        int k = static_cast<int>(floor(gz));

        // Get fractional part (interpolation weights)
        double dx = gx - i;
        double dy = gy - j;
        double dz = gz - k;

        // CIC weights for the 8 neighboring cells
        double wx[2] = {1.0 - dx, dx};
        double wy[2] = {1.0 - dy, dy};
        double wz[2] = {1.0 - dz, dz};

        // Interpolate forces from 8 neighboring grid points
        double forceX = 0.0, forceY = 0.0, forceZ = 0.0;

        for (int di = 0; di < 2; ++di)
        {
            for (int dj = 0; dj < 2; ++dj)
            {
                for (int dk = 0; dk < 2; ++dk)
                {
                    int gi = (i + di) % static_cast<int>(gridSize);
                    int gj = (j + dj) % static_cast<int>(gridSize);
                    int gk = (k + dk) % static_cast<int>(gridSize);

                    // Handle negative indices
                    if (gi < 0)
                        gi += gridSize;
                    if (gj < 0)
                        gj += gridSize;
                    if (gk < 0)
                        gk += gridSize;

                    double weight = wx[di] * wy[dj] * wz[dk];
                    size_t gridIndex = getGridIndex(gi, gj, gk);

                    forceX += hostForceX[gridIndex] * weight;
                    forceY += hostForceY[gridIndex] * weight;
                    forceZ += hostForceZ[gridIndex] * weight;
                }
            }
        }

        forces[p * 3] = forceX;
        forces[p * 3 + 1] = forceY;
        forces[p * 3 + 2] = forceZ;
    }
}

// Combined P3M operation
bool Mesh::computePMForces(const std::vector<double> &positions,
                           const std::vector<double> &masses,
                           std::vector<double> &forces,
                           size_t numParticles)
{
    // Step 1: Assign particles to mesh
    if (!assignParticlesToMesh(positions, masses, numParticles))
    {
        return false;
    }

    // Step 2: Solve Poisson equation
    if (!solvePoissonFFT())
    {
        return false;
    }

    // Step 3: Calculate forces from potential
    if (!calculateForcesFromPotential())
    {
        return false;
    }

    // Step 4: Interpolate forces to particles
    if (!interpolateForcesToParticles(positions, forces, numParticles))
    {
        return false;
    }

    return true;
}

// Reset density field
void Mesh::resetDensity()
{
    std::fill(hostDensity.begin(), hostDensity.end(), 0.0);
}

// Reset potential field
void Mesh::resetPotential()
{
    std::fill(hostPotential.begin(), hostPotential.end(), 0.0);
}

// Reset force fields
void Mesh::resetForces()
{
    std::fill(hostForceX.begin(), hostForceX.end(), 0.0);
    std::fill(hostForceY.begin(), hostForceY.end(), 0.0);
    std::fill(hostForceZ.begin(), hostForceZ.end(), 0.0);
}

// Get total mass on grid
double Mesh::getTotalMass() const
{
    double totalMass = 0.0;
    for (double density : hostDensity)
    {
        totalMass += density;
    }
    return totalMass;
}

// Get grid index from 3D coordinates
size_t Mesh::getGridIndex(size_t i, size_t j, size_t k) const
{
    return i * gridSize * gridSize + j * gridSize + k;
}

// Get 3D coordinates from grid index
void Mesh::getGridCoordinates(size_t index, size_t &i, size_t &j, size_t &k) const
{
    k = index % gridSize;
    size_t temp = index / gridSize;
    j = temp % gridSize;
    i = temp / gridSize;
}

// CIC weight function
double Mesh::cicWeight(double dx) const
{
    if (dx >= 0.0 && dx <= 1.0)
    {
        return 1.0 - dx;
    }
    else if (dx >= -1.0 && dx < 0.0)
    {
        return 1.0 + dx;
    }
    else
    {
        return 0.0;
    }
}

// Print grid information
void Mesh::printGridInfo() const
{
    std::cout << "=== Mesh Grid Information ===" << std::endl;
    std::cout << "Grid Size: " << gridSize << "x" << gridSize << "x" << gridSize << std::endl;
    std::cout << "Grid Spacing: " << gridSpacing << std::endl;
    std::cout << "Total Grid Points: " << getTotalGridPoints() << std::endl;
    std::cout << "Grid Origin: [" << gridOrigin[0] << ", " << gridOrigin[1] << ", " << gridOrigin[2] << "]" << std::endl;
    std::cout << "Total Mass: " << getTotalMass() << std::endl;
    std::cout << "FFT Plans Created: " << ((forwardPlan && inversePlan) ? "Yes" : "No") << std::endl;
    std::cout << "Device Memory Allocated: " << (isDeviceMemoryAllocated() ? "Yes" : "No") << std::endl;
    std::cout << "=============================" << std::endl;
}

// Green function for Poisson equation in Fourier space
double Mesh::greenFunction(double kx, double ky, double kz, double gridSpacing)
{
    double k2 = kx * kx + ky * ky + kz * kz;
    if (k2 == 0.0)
    {
        return 0.0; // Avoid division by zero
    }

    // In 3D, the Green function for -∇²Φ = 4πGρ is Φ(k) = -4πG ρ(k)/k²
    // For simplicity, we'll use 1/k² (ignoring constants)
    return -1.0 / k2;
}

// Data synchronization methods (simplified for now)
bool Mesh::copyDensityToDevice(CUDAManager &cudaMgr)
{
    if (!isDeviceMemoryAllocated())
        return false;
    // Implementation would copy hostDensity to deviceDensity
    return true;
}

bool Mesh::copyPotentialToHost(CUDAManager &cudaMgr)
{
    if (!isDeviceMemoryAllocated())
        return false;
    // Implementation would copy devicePotential to hostPotential
    return true;
}

bool Mesh::copyForcesToHost(CUDAManager &cudaMgr)
{
    if (!isDeviceMemoryAllocated())
        return false;
    // Implementation would copy device forces to host forces
    return true;
}