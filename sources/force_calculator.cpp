#include "force_calculator.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

// Constructor
ForceCalculator::ForceCalculator(ParticleSystem *particles, Mesh *mesh,
                                 CUDAManager *cudaMgr, double cutoffRadius,
                                 double ewaldAlpha)
    : particles(particles), mesh(mesh), cudaMgr(cudaMgr),
      cutoffRadius(cutoffRadius), ewaldAlpha(ewaldAlpha),
      softeningParameter(0.01), potentialEnergy(0.0), kineticEnergy(0.0),
      numForceEvaluations(0)
{

    if (!validateInputs())
    {
        throw std::runtime_error("Invalid inputs to ForceCalculator");
    }

    initializeForceArrays();
}

// Destructor
ForceCalculator::~ForceCalculator()
{
    // Force arrays are automatically cleaned up
}

// Validate input parameters
bool ForceCalculator::validateInputs() const
{
    if (!particles || !mesh || !cudaMgr)
    {
        std::cerr << "ForceCalculator: Null pointer inputs" << std::endl;
        return false;
    }

    if (cutoffRadius <= 0.0)
    {
        std::cerr << "ForceCalculator: Invalid cutoff radius" << std::endl;
        return false;
    }

    if (ewaldAlpha <= 0.0)
    {
        std::cerr << "ForceCalculator: Invalid Ewald alpha" << std::endl;
        return false;
    }

    return true;
}

// Initialize force arrays
void ForceCalculator::initializeForceArrays()
{
    size_t numParticles = particles->getNumParticles();
    size_t arraySize = numParticles * 3; // 3 components per particle

    ppForces.resize(arraySize, 0.0);
    pmForces.resize(arraySize, 0.0);
    totalForces.resize(arraySize, 0.0);
}

// Main force calculation method
bool ForceCalculator::calculateForces(ForceMethod method)
{
    if (!validateInputs())
    {
        return false;
    }

    resetForces();
    numForceEvaluations++;

    bool success = false;

    switch (method)
    {
    case ForceMethod::DIRECT_SUMMATION:
        success = calculateDirectForces();
        break;

    case ForceMethod::PARTICLE_MESH:
        success = calculatePMForces();
        break;

    case ForceMethod::PARTICLE_PARTICLE:
        success = calculatePPForces();
        break;

    case ForceMethod::P3M_FULL:
        success = calculatePPForces() && calculatePMForces() && combineForces();
        break;

    default:
        std::cerr << "ForceCalculator: Unknown force method" << std::endl;
        return false;
    }

    if (success && method != ForceMethod::DIRECT_SUMMATION)
    {
        success &= applyForcesToParticles();
    }

    return success;
}

// Calculate particle-particle forces
bool ForceCalculator::calculatePPForces()
{
    if (cudaMgr->isInitialized())
    {
        return calculatePPForcesGPU();
    }
    else
    {
        return calculatePPForcesCPU();
    }
}

// CPU implementation of particle-particle forces
bool ForceCalculator::calculatePPForcesCPU()
{
    const auto &positions = particles->getHostPositions();
    const auto &masses = particles->getHostMasses();
    size_t numParticles = particles->getNumParticles();

    potentialEnergy = 0.0;

    // Reset PP forces
    std::fill(ppForces.begin(), ppForces.end(), 0.0);

    // Calculate forces between all pairs within cutoff
    for (size_t i = 0; i < numParticles; ++i)
    {
        for (size_t j = i + 1; j < numParticles; ++j)
        {
            const double *pos1 = &positions[i * 3];
            const double *pos2 = &positions[j * 3];

            double distance = calculateDistance(pos1, pos2);

            // Apply cutoff
            if (distance > cutoffRadius)
            {
                continue;
            }

            // Calculate force components
            double force[3];
            double pairPotential;
            calculateForceComponents(pos1, pos2, masses[i], masses[j],
                                     force, pairPotential);

            // Add forces (Newton's 3rd law: F_ij = -F_ji)
            for (int k = 0; k < 3; ++k)
            {
                ppForces[i * 3 + k] += force[k];
                ppForces[j * 3 + k] -= force[k];
            }

            potentialEnergy += pairPotential;
        }
    }

    return true;
}

// GPU implementation of particle-particle forces (simplified for now)
bool ForceCalculator::calculatePPForcesGPU()
{
    // For now, fall back to CPU implementation
    // In a full implementation, this would use CUDA kernels
    std::cout << "GPU PP forces not implemented yet, using CPU" << std::endl;
    return calculatePPForcesCPU();
}

// Calculate particle-mesh forces
bool ForceCalculator::calculatePMForces()
{
    const auto &positions = particles->getHostPositions();
    const auto &masses = particles->getHostMasses();
    size_t numParticles = particles->getNumParticles();

    // Step 1: Assign particles to mesh
    if (!mesh->assignParticlesToMesh(positions, masses, numParticles))
    {
        std::cerr << "Failed to assign particles to mesh" << std::endl;
        return false;
    }

    // Step 2: Solve Poisson equation (simplified - would use FFT in full implementation)
    if (!mesh->solvePoissonFFT())
    {
        std::cerr << "Failed to solve Poisson equation" << std::endl;
        return false;
    }

    // Step 3: Calculate forces from potential
    if (!mesh->calculateForcesFromPotential())
    {
        std::cerr << "Failed to calculate forces from potential" << std::endl;
        return false;
    }

    // Step 4: Interpolate forces to particles
    if (!mesh->interpolateForcesToParticles(positions, pmForces, numParticles))
    {
        std::cerr << "Failed to interpolate forces to particles" << std::endl;
        return false;
    }

    return true;
}

// Calculate direct summation forces (for validation)
bool ForceCalculator::calculateDirectForces()
{
    return calculateDirectForcesCPU();
}

// CPU direct summation implementation
bool ForceCalculator::calculateDirectForcesCPU()
{
    const auto &positions = particles->getHostPositions();
    const auto &masses = particles->getHostMasses();
    size_t numParticles = particles->getNumParticles();

    potentialEnergy = 0.0;

    // Reset total forces (using totalForces as direct forces)
    std::fill(totalForces.begin(), totalForces.end(), 0.0);

    // Calculate forces between all pairs (no cutoff for direct summation)
    for (size_t i = 0; i < numParticles; ++i)
    {
        for (size_t j = i + 1; j < numParticles; ++j)
        {
            const double *pos1 = &positions[i * 3];
            const double *pos2 = &positions[j * 3];

            double distance = calculateDistance(pos1, pos2);

            // Avoid division by zero
            if (distance < softeningParameter)
            {
                distance = softeningParameter;
            }

            // Calculate gravitational force: F = G * m1 * m2 / r^2
            double forceMagnitude = masses[i] * masses[j] / (distance * distance);

            // Calculate force components
            double dx = pos2[0] - pos1[0];
            double dy = pos2[1] - pos1[1];
            double dz = pos2[2] - pos1[2];

            // Normalize direction
            double invDistance = 1.0 / distance;
            dx *= invDistance;
            dy *= invDistance;
            dz *= invDistance;

            // Apply force magnitude
            double forceX = forceMagnitude * dx;
            double forceY = forceMagnitude * dy;
            double forceZ = forceMagnitude * dz;

            // Add forces (Newton's 3rd law)
            totalForces[i * 3] += forceX;
            totalForces[i * 3 + 1] += forceY;
            totalForces[i * 3 + 2] += forceZ;

            totalForces[j * 3] -= forceX;
            totalForces[j * 3 + 1] -= forceY;
            totalForces[j * 3 + 2] -= forceZ;

            // Calculate potential energy
            potentialEnergy += -masses[i] * masses[j] / distance;
        }
    }

    return true;
}

// Combine PP and PM forces
bool ForceCalculator::combineForces()
{
    size_t arraySize = ppForces.size();

    if (pmForces.size() != arraySize || totalForces.size() != arraySize)
    {
        std::cerr << "Force array size mismatch" << std::endl;
        return false;
    }

    for (size_t i = 0; i < arraySize; ++i)
    {
        totalForces[i] = ppForces[i] + pmForces[i];
    }

    return true;
}

// Apply forces to particles
bool ForceCalculator::applyForcesToParticles()
{
    return particles->setForces(totalForces);
}

// Calculate distance between two positions
double ForceCalculator::calculateDistance(const double *pos1, const double *pos2) const
{
    double dx = pos2[0] - pos1[0];
    double dy = pos2[1] - pos1[1];
    double dz = pos2[2] - pos1[2];

    // Apply periodic boundary conditions if needed
    // For now, assume non-periodic

    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Calculate force components between two particles
void ForceCalculator::calculateForceComponents(const double *pos1, const double *pos2,
                                               double mass1, double mass2,
                                               double *force, double &potential) const
{
    double dx = pos2[0] - pos1[0];
    double dy = pos2[1] - pos1[1];
    double dz = pos2[2] - pos1[2];

    double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

    // Avoid division by zero
    if (distance < softeningParameter)
    {
        distance = softeningParameter;
    }

    // Calculate force magnitude with Ewald short-range screening
    double forceMagnitude = mass1 * mass2 * ewaldShortRangeForce(distance, ewaldAlpha);

    // Normalize direction
    double invDistance = 1.0 / distance;
    dx *= invDistance;
    dy *= invDistance;
    dz *= invDistance;

    // Apply force components
    force[0] = forceMagnitude * dx;
    force[1] = forceMagnitude * dy;
    force[2] = forceMagnitude * dz;

    // Calculate potential
    potential = mass1 * mass2 * ewaldShortRangePotential(distance, ewaldAlpha);
}

// Ewald short-range potential
double ForceCalculator::ewaldShortRangePotential(double r, double alpha) const
{
    if (r == 0.0)
        return 0.0;

    // Complementary error function for Ewald screening
    double alpha_r = alpha * r;
    double erfc_val = erfc(alpha_r);

    return erfc_val / r;
}

// Ewald short-range force
double ForceCalculator::ewaldShortRangeForce(double r, double alpha) const
{
    if (r == 0.0)
        return 0.0;

    double alpha_r = alpha * r;
    double erfc_val = erfc(alpha_r);
    double exp_val = exp(-alpha_r * alpha_r);

    // Force = -dΦ/dr = -d/dr(erfc(αr)/r)
    double term1 = erfc_val / (r * r);
    double term2 = (2.0 * alpha / sqrt(M_PI)) * exp_val / r;

    return term1 - term2;
}

// Complementary error function (simplified implementation)
double ForceCalculator::erfcEwald(double x, double alpha) const
{
    // For simplicity, use the standard erfc function
    // In a full implementation, you might want a more optimized version
    return erfc(x);
}

// Compare P3M with direct summation
bool ForceCalculator::compareWithDirectSummation(double &maxError, double &rmsError,
                                                 size_t numTestParticles)
{
    // Limit test particles for performance
    size_t actualTestParticles = std::min(numTestParticles, particles->getNumParticles());

    // Calculate direct summation forces
    if (!calculateDirectForces())
    {
        std::cerr << "Failed to calculate direct forces for comparison" << std::endl;
        return false;
    }

    // Store direct forces
    std::vector<double> directForces = totalForces;

    // Calculate P3M forces
    if (!calculateForces(ForceMethod::P3M_FULL))
    {
        std::cerr << "Failed to calculate P3M forces for comparison" << std::endl;
        return false;
    }

    // Compare forces
    maxError = 0.0;
    rmsError = 0.0;
    size_t numComparisons = 0;

    for (size_t i = 0; i < actualTestParticles; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            size_t idx = i * 3 + k;
            double p3mForce = totalForces[idx];
            double directForce = directForces[idx];

            double error = std::abs(p3mForce - directForce);
            maxError = std::max(maxError, error);
            rmsError += error * error;
            numComparisons++;
        }
    }

    if (numComparisons > 0)
    {
        rmsError = std::sqrt(rmsError / numComparisons);
    }

    return true;
}

// Reset all force arrays
void ForceCalculator::resetForces()
{
    std::fill(ppForces.begin(), ppForces.end(), 0.0);
    std::fill(pmForces.begin(), pmForces.end(), 0.0);
    std::fill(totalForces.begin(), totalForces.end(), 0.0);
    potentialEnergy = 0.0;
    kineticEnergy = 0.0;
}

// Setters
void ForceCalculator::setCutoffRadius(double cutoff)
{
    if (cutoff > 0.0)
    {
        cutoffRadius = cutoff;
    }
}

void ForceCalculator::setEwaldAlpha(double alpha)
{
    if (alpha > 0.0)
    {
        ewaldAlpha = alpha;
    }
}

void ForceCalculator::setSofteningParameter(double softening)
{
    if (softening >= 0.0)
    {
        softeningParameter = softening;
    }
}

// Getters
double ForceCalculator::getPotentialEnergy() const
{
    return potentialEnergy;
}

double ForceCalculator::getKineticEnergy() const
{
    // Calculate kinetic energy from particle velocities
    const auto &velocities = particles->getHostVelocities();
    const auto &masses = particles->getHostMasses();
    size_t numParticles = particles->getNumParticles();

    double ke = 0.0;
    for (size_t i = 0; i < numParticles; ++i)
    {
        double vx = velocities[i * 3];
        double vy = velocities[i * 3 + 1];
        double vz = velocities[i * 3 + 2];
        double mass = masses[i];

        ke += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
    }

    return ke;
}

// Print performance statistics
void ForceCalculator::printPerformanceStats() const
{
    std::cout << "=== Force Calculator Performance ===" << std::endl;
    std::cout << "Force evaluations: " << numForceEvaluations << std::endl;
    std::cout << "Potential energy: " << potentialEnergy << std::endl;
    std::cout << "Kinetic energy: " << getKineticEnergy() << std::endl;
    std::cout << "Total energy: " << potentialEnergy + getKineticEnergy() << std::endl;
    std::cout << "====================================" << std::endl;
}

// Placeholder for GPU kernel (would be in .cu file)
void ForceCalculator::computePPForcesKernel(const double *positions, const double *masses,
                                            double *forces, size_t numParticles,
                                            double cutoff, double alpha, double softening)
{
    // This would be implemented as a CUDA kernel
    // For now, it's a placeholder
    std::cout << "GPU kernel not implemented - using CPU fallback" << std::endl;
}