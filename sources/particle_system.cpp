#include "particle_system.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

// Constructor
ParticleSystem::ParticleSystem(size_t numParticles)
    : numParticles(numParticles)
{
    initializeHostData();
}

// Destructor
ParticleSystem::~ParticleSystem()
{
    // Host data is automatically cleaned up by std::vector
    // Device memory should be explicitly freed by calling freeDeviceMemory()
}

// Initialize host data structures
void ParticleSystem::initializeHostData()
{
    hostPositions.resize(numParticles * 3, 0.0);
    hostVelocities.resize(numParticles * 3, 0.0);
    hostMasses.resize(numParticles, 1.0);
    hostForces.resize(numParticles * 3, 0.0);
    hostPotentials.resize(numParticles, 0.0);
}

// Allocate host memory (if needed for large datasets)
bool ParticleSystem::allocateHostMemory()
{
    try
    {
        // Vectors automatically allocate memory
        initializeHostData();
        return true;
    }
    catch (const std::bad_alloc &e)
    {
        std::cerr << "Failed to allocate host memory: " << e.what() << std::endl;
        return false;
    }
}

// Initialize particles with random positions and velocities
bool ParticleSystem::initializeRandom(CUDAManager &cudaMgr, double boxSize)
{
    if (!allocateHostMemory())
    {
        return false;
    }

    // Generate random positions within a box
    generateRandomPositions(boxSize);

    // Generate random velocities
    generateRandomVelocities(0.1 * boxSize);

    // Generate uniform masses
    generateUniformMasses(1.0);

    // Reset forces and potentials
    resetForces();
    resetPotentials();

    // Copy to device if CUDA manager is available
    if (cudaMgr.isInitialized())
    {
        return copyToDevice(cudaMgr);
    }

    return true;
}

// Initialize particles on a uniform grid
bool ParticleSystem::initializeUniformGrid(CUDAManager &cudaMgr, double spacing)
{
    if (!allocateHostMemory())
    {
        return false;
    }

    // Calculate grid dimensions
    size_t gridSize = static_cast<size_t>(std::ceil(std::cbrt(numParticles)));
    double offset = -0.5 * (gridSize - 1) * spacing;

    size_t particleIndex = 0;
    for (size_t i = 0; i < gridSize && particleIndex < numParticles; ++i)
    {
        for (size_t j = 0; j < gridSize && particleIndex < numParticles; ++j)
        {
            for (size_t k = 0; k < gridSize && particleIndex < numParticles; ++k)
            {
                size_t posIndex = particleIndex * 3;
                hostPositions[posIndex] = offset + i * spacing;
                hostPositions[posIndex + 1] = offset + j * spacing;
                hostPositions[posIndex + 2] = offset + k * spacing;

                ++particleIndex;
            }
        }
    }

    // Generate small random velocities to break symmetry
    generateRandomVelocities(0.01 * spacing);

    // Generate uniform masses
    generateUniformMasses(1.0);

    // Reset forces and potentials
    resetForces();
    resetPotentials();

    // Copy to device if CUDA manager is available
    if (cudaMgr.isInitialized())
    {
        return copyToDevice(cudaMgr);
    }

    return true;
}

// Initialize from file
bool ParticleSystem::initializeFromFile(CUDAManager &cudaMgr, const std::string &filename)
{
    if (!loadFromFile(filename))
    {
        return false;
    }

    // Copy to device if CUDA manager is available
    if (cudaMgr.isInitialized())
    {
        return copyToDevice(cudaMgr);
    }

    return true;
}

// Generate random positions within a box
void ParticleSystem::generateRandomPositions(double boxSize)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-boxSize / 2.0, boxSize / 2.0);

    for (size_t i = 0; i < numParticles * 3; ++i)
    {
        hostPositions[i] = dis(gen);
    }
}

// Generate random velocities
void ParticleSystem::generateRandomVelocities(double maxVelocity)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-maxVelocity, maxVelocity);

    for (size_t i = 0; i < numParticles * 3; ++i)
    {
        hostVelocities[i] = dis(gen);
    }
}

// Generate uniform masses
void ParticleSystem::generateUniformMasses(double mass)
{
    std::fill(hostMasses.begin(), hostMasses.end(), mass);
}

// Set positions
bool ParticleSystem::setPositions(const std::vector<double> &positions)
{
    if (!validateArraySize(positions, numParticles * 3))
    {
        return false;
    }
    hostPositions = positions;
    return true;
}

// Set velocities
bool ParticleSystem::setVelocities(const std::vector<double> &velocities)
{
    if (!validateArraySize(velocities, numParticles * 3))
    {
        return false;
    }
    hostVelocities = velocities;
    return true;
}

// Set masses
bool ParticleSystem::setMasses(const std::vector<double> &masses)
{
    if (!validateArraySize(masses, numParticles))
    {
        return false;
    }
    hostMasses = masses;
    return true;
}

// Set forces
bool ParticleSystem::setForces(const std::vector<double> &forces)
{
    if (!validateArraySize(forces, numParticles * 3))
    {
        return false;
    }
    hostForces = forces;
    return true;
}

// Set potentials
bool ParticleSystem::setPotentials(const std::vector<double> &potentials)
{
    if (!validateArraySize(potentials, numParticles))
    {
        return false;
    }
    hostPotentials = potentials;
    return true;
}

// Reset forces to zero
void ParticleSystem::resetForces()
{
    std::fill(hostForces.begin(), hostForces.end(), 0.0);
}

// Reset potentials to zero
void ParticleSystem::resetPotentials()
{
    std::fill(hostPotentials.begin(), hostPotentials.end(), 0.0);
}

// Allocate device memory
bool ParticleSystem::allocateDeviceMemory(CUDAManager &cudaMgr)
{
    if (!cudaMgr.isInitialized())
    {
        std::cerr << "CUDA manager not initialized" << std::endl;
        return false;
    }

    size_t vectorSize = numParticles * sizeof(double);
    size_t vector3Size = numParticles * 3 * sizeof(double);

    // Allocate device memory for each array
    bool success = true;
    success &= cudaMgr.allocateDeviceMemory(&deviceArrays.positions_x, vectorSize);
    success &= cudaMgr.allocateDeviceMemory(&deviceArrays.positions_y, vectorSize);
    success &= cudaMgr.allocateDeviceMemory(&deviceArrays.positions_z, vectorSize);
    success &= cudaMgr.allocateDeviceMemory(&deviceArrays.velocities_x, vectorSize);
    success &= cudaMgr.allocateDeviceMemory(&deviceArrays.velocities_y, vectorSize);
    success &= cudaMgr.allocateDeviceMemory(&deviceArrays.velocities_z, vectorSize);
    success &= cudaMgr.allocateDeviceMemory(&deviceArrays.masses, vectorSize);
    success &= cudaMgr.allocateDeviceMemory(&deviceArrays.forces_x, vectorSize);
    success &= cudaMgr.allocateDeviceMemory(&deviceArrays.forces_y, vectorSize);
    success &= cudaMgr.allocateDeviceMemory(&deviceArrays.forces_z, vectorSize);
    success &= cudaMgr.allocateDeviceMemory(&deviceArrays.potentials, vectorSize);

    if (!success)
    {
        std::cerr << "Failed to allocate device memory for particle system" << std::endl;
        freeDeviceMemory(cudaMgr);
        return false;
    }

    return true;
}

// Free device memory
bool ParticleSystem::freeDeviceMemory(CUDAManager &cudaMgr)
{
    if (!cudaMgr.isInitialized())
    {
        return true; // Nothing to free
    }

    bool success = true;
    success &= cudaMgr.freeDeviceMemory(deviceArrays.positions_x);
    success &= cudaMgr.freeDeviceMemory(deviceArrays.positions_y);
    success &= cudaMgr.freeDeviceMemory(deviceArrays.positions_z);
    success &= cudaMgr.freeDeviceMemory(deviceArrays.velocities_x);
    success &= cudaMgr.freeDeviceMemory(deviceArrays.velocities_y);
    success &= cudaMgr.freeDeviceMemory(deviceArrays.velocities_z);
    success &= cudaMgr.freeDeviceMemory(deviceArrays.masses);
    success &= cudaMgr.freeDeviceMemory(deviceArrays.forces_x);
    success &= cudaMgr.freeDeviceMemory(deviceArrays.forces_y);
    success &= cudaMgr.freeDeviceMemory(deviceArrays.forces_z);
    success &= cudaMgr.freeDeviceMemory(deviceArrays.potentials);

    // Reset pointers
    deviceArrays = ParticleArrays();

    return success;
}

// Copy all data to device
bool ParticleSystem::copyToDevice(CUDAManager &cudaMgr)
{
    if (!isDeviceMemoryAllocated())
    {
        if (!allocateDeviceMemory(cudaMgr))
        {
            return false;
        }
    }

    return copyPositionsToDevice(cudaMgr) &&
           copyVelocitiesToDevice(cudaMgr) &&
           copyMassesToDevice(cudaMgr) &&
           copyForcesToDevice(cudaMgr) &&
           copyPotentialsToDevice(cudaMgr);
}

// Copy positions to device
bool ParticleSystem::copyPositionsToDevice(CUDAManager &cudaMgr)
{
    if (!isDeviceMemoryAllocated())
    {
        return false;
    }

    // Extract individual coordinate arrays
    std::vector<double> posX(numParticles), posY(numParticles), posZ(numParticles);
    for (size_t i = 0; i < numParticles; ++i)
    {
        posX[i] = hostPositions[i * 3];
        posY[i] = hostPositions[i * 3 + 1];
        posZ[i] = hostPositions[i * 3 + 2];
    }

    return cudaMgr.copyHostToDevice(posX.data(), deviceArrays.positions_x, numParticles * sizeof(double)) &&
           cudaMgr.copyHostToDevice(posY.data(), deviceArrays.positions_y, numParticles * sizeof(double)) &&
           cudaMgr.copyHostToDevice(posZ.data(), deviceArrays.positions_z, numParticles * sizeof(double));
}

// Copy velocities to device
bool ParticleSystem::copyVelocitiesToDevice(CUDAManager &cudaMgr)
{
    if (!isDeviceMemoryAllocated())
    {
        return false;
    }

    // Extract individual coordinate arrays
    std::vector<double> velX(numParticles), velY(numParticles), velZ(numParticles);
    for (size_t i = 0; i < numParticles; ++i)
    {
        velX[i] = hostVelocities[i * 3];
        velY[i] = hostVelocities[i * 3 + 1];
        velZ[i] = hostVelocities[i * 3 + 2];
    }

    return cudaMgr.copyHostToDevice(velX.data(), deviceArrays.velocities_x, numParticles * sizeof(double)) &&
           cudaMgr.copyHostToDevice(velY.data(), deviceArrays.velocities_y, numParticles * sizeof(double)) &&
           cudaMgr.copyHostToDevice(velZ.data(), deviceArrays.velocities_z, numParticles * sizeof(double));
}

// Copy masses to device
bool ParticleSystem::copyMassesToDevice(CUDAManager &cudaMgr)
{
    if (!isDeviceMemoryAllocated())
    {
        return false;
    }

    return cudaMgr.copyHostToDevice(hostMasses.data(), deviceArrays.masses, numParticles * sizeof(double));
}

// Copy forces to device
bool ParticleSystem::copyForcesToDevice(CUDAManager &cudaMgr)
{
    if (!isDeviceMemoryAllocated())
    {
        return false;
    }

    // Extract individual coordinate arrays
    std::vector<double> forceX(numParticles), forceY(numParticles), forceZ(numParticles);
    for (size_t i = 0; i < numParticles; ++i)
    {
        forceX[i] = hostForces[i * 3];
        forceY[i] = hostForces[i * 3 + 1];
        forceZ[i] = hostForces[i * 3 + 2];
    }

    return cudaMgr.copyHostToDevice(forceX.data(), deviceArrays.forces_x, numParticles * sizeof(double)) &&
           cudaMgr.copyHostToDevice(forceY.data(), deviceArrays.forces_y, numParticles * sizeof(double)) &&
           cudaMgr.copyHostToDevice(forceZ.data(), deviceArrays.forces_z, numParticles * sizeof(double));
}

// Copy potentials to device
bool ParticleSystem::copyPotentialsToDevice(CUDAManager &cudaMgr)
{
    if (!isDeviceMemoryAllocated())
    {
        return false;
    }

    return cudaMgr.copyHostToDevice(hostPotentials.data(), deviceArrays.potentials, numParticles * sizeof(double));
}

// Copy all data to host
bool ParticleSystem::copyToHost(CUDAManager &cudaMgr)
{
    if (!isDeviceMemoryAllocated())
    {
        return false;
    }

    // Copy positions
    std::vector<double> posX(numParticles), posY(numParticles), posZ(numParticles);
    bool success = cudaMgr.copyDeviceToHost(deviceArrays.positions_x, posX.data(), numParticles * sizeof(double));
    success &= cudaMgr.copyDeviceToHost(deviceArrays.positions_y, posY.data(), numParticles * sizeof(double));
    success &= cudaMgr.copyDeviceToHost(deviceArrays.positions_z, posZ.data(), numParticles * sizeof(double));

    if (success)
    {
        for (size_t i = 0; i < numParticles; ++i)
        {
            hostPositions[i * 3] = posX[i];
            hostPositions[i * 3 + 1] = posY[i];
            hostPositions[i * 3 + 2] = posZ[i];
        }
    }

    // Copy velocities
    std::vector<double> velX(numParticles), velY(numParticles), velZ(numParticles);
    success &= cudaMgr.copyDeviceToHost(deviceArrays.velocities_x, velX.data(), numParticles * sizeof(double));
    success &= cudaMgr.copyDeviceToHost(deviceArrays.velocities_y, velY.data(), numParticles * sizeof(double));
    success &= cudaMgr.copyDeviceToHost(deviceArrays.velocities_z, velZ.data(), numParticles * sizeof(double));

    if (success)
    {
        for (size_t i = 0; i < numParticles; ++i)
        {
            hostVelocities[i * 3] = velX[i];
            hostVelocities[i * 3 + 1] = velY[i];
            hostVelocities[i * 3 + 2] = velZ[i];
        }
    }

    // Copy masses
    success &= cudaMgr.copyDeviceToHost(deviceArrays.masses, hostMasses.data(), numParticles * sizeof(double));

    // Copy forces
    std::vector<double> forceX(numParticles), forceY(numParticles), forceZ(numParticles);
    success &= cudaMgr.copyDeviceToHost(deviceArrays.forces_x, forceX.data(), numParticles * sizeof(double));
    success &= cudaMgr.copyDeviceToHost(deviceArrays.forces_y, forceY.data(), numParticles * sizeof(double));
    success &= cudaMgr.copyDeviceToHost(deviceArrays.forces_z, forceZ.data(), numParticles * sizeof(double));

    if (success)
    {
        for (size_t i = 0; i < numParticles; ++i)
        {
            hostForces[i * 3] = forceX[i];
            hostForces[i * 3 + 1] = forceY[i];
            hostForces[i * 3 + 2] = forceZ[i];
        }
    }

    // Copy potentials
    success &= cudaMgr.copyDeviceToHost(deviceArrays.potentials, hostPotentials.data(), numParticles * sizeof(double));

    return success;
}

// Get total mass
double ParticleSystem::getTotalMass() const
{
    return std::accumulate(hostMasses.begin(), hostMasses.end(), 0.0);
}

// Get center of mass
void ParticleSystem::getCenterOfMass(double *com) const
{
    calculateCenterOfMass(com);
}

// Get total momentum
void ParticleSystem::getTotalMomentum(double *momentum) const
{
    calculateTotalMomentum(momentum);
}

// Get kinetic energy
double ParticleSystem::getKineticEnergy() const
{
    double kineticEnergy = 0.0;
    for (size_t i = 0; i < numParticles; ++i)
    {
        double vx = hostVelocities[i * 3];
        double vy = hostVelocities[i * 3 + 1];
        double vz = hostVelocities[i * 3 + 2];
        double mass = hostMasses[i];
        kineticEnergy += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
    }
    return kineticEnergy;
}

// Get potential energy
double ParticleSystem::getPotentialEnergy() const
{
    return std::accumulate(hostPotentials.begin(), hostPotentials.end(), 0.0);
}

// Calculate center of mass
void ParticleSystem::calculateCenterOfMass(double *com) const
{
    double totalMass = getTotalMass();
    if (totalMass == 0.0)
    {
        com[0] = com[1] = com[2] = 0.0;
        return;
    }

    com[0] = com[1] = com[2] = 0.0;
    for (size_t i = 0; i < numParticles; ++i)
    {
        double mass = hostMasses[i];
        com[0] += mass * hostPositions[i * 3];
        com[1] += mass * hostPositions[i * 3 + 1];
        com[2] += mass * hostPositions[i * 3 + 2];
    }

    com[0] /= totalMass;
    com[1] /= totalMass;
    com[2] /= totalMass;
}

// Calculate total momentum
void ParticleSystem::calculateTotalMomentum(double *momentum) const
{
    momentum[0] = momentum[1] = momentum[2] = 0.0;
    for (size_t i = 0; i < numParticles; ++i)
    {
        double mass = hostMasses[i];
        momentum[0] += mass * hostVelocities[i * 3];
        momentum[1] += mass * hostVelocities[i * 3 + 1];
        momentum[2] += mass * hostVelocities[i * 3 + 2];
    }
}

// Validate array size
bool ParticleSystem::validateArraySize(const std::vector<double> &array, size_t expectedSize) const
{
    if (array.size() != expectedSize)
    {
        std::cerr << "Array size mismatch. Expected: " << expectedSize
                  << ", Got: " << array.size() << std::endl;
        return false;
    }
    return true;
}

// Validate data integrity
bool ParticleSystem::validateData() const
{
    if (hostPositions.size() != numParticles * 3)
        return false;
    if (hostVelocities.size() != numParticles * 3)
        return false;
    if (hostMasses.size() != numParticles)
        return false;
    if (hostForces.size() != numParticles * 3)
        return false;
    if (hostPotentials.size() != numParticles)
        return false;

    // Check for NaN or infinite values
    for (const auto &val : hostPositions)
    {
        if (std::isnan(val) || std::isinf(val))
            return false;
    }
    for (const auto &val : hostVelocities)
    {
        if (std::isnan(val) || std::isinf(val))
            return false;
    }
    for (const auto &val : hostMasses)
    {
        if (std::isnan(val) || std::isinf(val) || val <= 0.0)
            return false;
    }

    return true;
}

// Print system summary
void ParticleSystem::printSummary() const
{
    std::cout << "=== Particle System Summary ===" << std::endl;
    std::cout << "Number of particles: " << numParticles << std::endl;
    std::cout << "Total mass: " << getTotalMass() << std::endl;

    double com[3];
    getCenterOfMass(com);
    std::cout << "Center of mass: [" << com[0] << ", " << com[1] << ", " << com[2] << "]" << std::endl;

    double momentum[3];
    getTotalMomentum(momentum);
    std::cout << "Total momentum: [" << momentum[0] << ", " << momentum[1] << ", " << momentum[2] << "]" << std::endl;

    std::cout << "Kinetic energy: " << getKineticEnergy() << std::endl;
    std::cout << "Potential energy: " << getPotentialEnergy() << std::endl;
    std::cout << "Total energy: " << getKineticEnergy() + getPotentialEnergy() << std::endl;

    std::cout << "Device memory allocated: " << (isDeviceMemoryAllocated() ? "Yes" : "No") << std::endl;
    std::cout << "Data valid: " << (validateData() ? "Yes" : "No") << std::endl;
    std::cout << "===============================" << std::endl;
}

// Save to file
bool ParticleSystem::saveToFile(const std::string &filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    file << "# Particle System Data\n";
    file << "# Format: x y z vx vy vz mass force_x force_y force_z potential\n";
    file << numParticles << "\n";

    for (size_t i = 0; i < numParticles; ++i)
    {
        size_t posIdx = i * 3;
        file << hostPositions[posIdx] << " " << hostPositions[posIdx + 1] << " " << hostPositions[posIdx + 2] << " ";
        file << hostVelocities[posIdx] << " " << hostVelocities[posIdx + 1] << " " << hostVelocities[posIdx + 2] << " ";
        file << hostMasses[i] << " ";
        file << hostForces[posIdx] << " " << hostForces[posIdx + 1] << " " << hostForces[posIdx + 2] << " ";
        file << hostPotentials[i] << "\n";
    }

    file.close();
    return true;
}

// Load from file
bool ParticleSystem::loadFromFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Cannot open file for reading: " << filename << std::endl;
        return false;
    }

    std::string line;
    size_t fileNumParticles = 0;

    // Read header
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        iss >> fileNumParticles;
        break;
    }

    if (fileNumParticles != numParticles)
    {
        std::cerr << "Particle count mismatch. Expected: " << numParticles
                  << ", File: " << fileNumParticles << std::endl;
        return false;
    }

    // Read particle data
    for (size_t i = 0; i < numParticles; ++i)
    {
        if (!std::getline(file, line))
        {
            std::cerr << "Unexpected end of file at particle " << i << std::endl;
            return false;
        }

        std::istringstream iss(line);
        size_t posIdx = i * 3;

        iss >> hostPositions[posIdx] >> hostPositions[posIdx + 1] >> hostPositions[posIdx + 2];
        iss >> hostVelocities[posIdx] >> hostVelocities[posIdx + 1] >> hostVelocities[posIdx + 2];
        iss >> hostMasses[i];
        iss >> hostForces[posIdx] >> hostForces[posIdx + 1] >> hostForces[posIdx + 2];
        iss >> hostPotentials[i];

        if (iss.fail())
        {
            std::cerr << "Error parsing data for particle " << i << std::endl;
            return false;
        }
    }

    file.close();
    return validateData();
}