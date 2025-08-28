#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "cuda_manager.h"
#include <vector>
#include <memory>

// Particle data structure
struct Particle
{
    double position[3]; // x, y, z coordinates
    double velocity[3]; // vx, vy, vz components
    double mass;        // particle mass
    double force[3];    // accumulated force components
    double potential;   // gravitational potential

    // Constructor
    Particle() : mass(1.0), potential(0.0)
    {
        for (int i = 0; i < 3; ++i)
        {
            position[i] = 0.0;
            velocity[i] = 0.0;
            force[i] = 0.0;
        }
    }
};

// GPU memory layout (Structure of Arrays for better coalescing)
struct ParticleArrays
{
    double *positions_x;
    double *positions_y;
    double *positions_z;
    double *velocities_x;
    double *velocities_y;
    double *velocities_z;
    double *masses;
    double *forces_x;
    double *forces_y;
    double *forces_z;
    double *potentials;

    // Constructor
    ParticleArrays() : positions_x(nullptr), positions_y(nullptr), positions_z(nullptr),
                       velocities_x(nullptr), velocities_y(nullptr), velocities_z(nullptr),
                       masses(nullptr), forces_x(nullptr), forces_y(nullptr), forces_z(nullptr),
                       potentials(nullptr) {}

    // Check if all arrays are allocated
    bool isAllocated() const
    {
        return positions_x && positions_y && positions_z &&
               velocities_x && velocities_y && velocities_z &&
               masses && forces_x && forces_y && forces_z && potentials;
    }
};

// Particle system class for managing N-body particle data
class ParticleSystem
{
public:
    // Constructor and destructor
    ParticleSystem(size_t numParticles);
    ~ParticleSystem();

    // Initialization methods
    bool initializeRandom(CUDAManager &cudaMgr, double boxSize = 1.0);
    bool initializeUniformGrid(CUDAManager &cudaMgr, double spacing = 1.0);
    bool initializeFromFile(CUDAManager &cudaMgr, const std::string &filename);

    // Data access methods
    size_t getNumParticles() const { return numParticles; }

    // Host data access
    const std::vector<double> &getHostPositions() const { return hostPositions; }
    const std::vector<double> &getHostVelocities() const { return hostVelocities; }
    const std::vector<double> &getHostMasses() const { return hostMasses; }
    const std::vector<double> &getHostForces() const { return hostForces; }
    const std::vector<double> &getHostPotentials() const { return hostPotentials; }

    // Device data access
    double *getDevicePositionsX() const { return deviceArrays.positions_x; }
    double *getDevicePositionsY() const { return deviceArrays.positions_y; }
    double *getDevicePositionsZ() const { return deviceArrays.positions_z; }
    double *getDeviceVelocitiesX() const { return deviceArrays.velocities_x; }
    double *getDeviceVelocitiesY() const { return deviceArrays.velocities_y; }
    double *getDeviceVelocitiesZ() const { return deviceArrays.velocities_z; }
    double *getDeviceMasses() const { return deviceArrays.masses; }
    double *getDeviceForcesX() const { return deviceArrays.forces_x; }
    double *getDeviceForcesY() const { return deviceArrays.forces_y; }
    double *getDeviceForcesZ() const { return deviceArrays.forces_z; }
    double *getDevicePotentials() const { return deviceArrays.potentials; }

    // Data modification methods
    bool setPositions(const std::vector<double> &positions);
    bool setVelocities(const std::vector<double> &velocities);
    bool setMasses(const std::vector<double> &masses);
    bool setForces(const std::vector<double> &forces);
    bool setPotentials(const std::vector<double> &potentials);

    // Data synchronization
    bool copyToDevice(CUDAManager &cudaMgr);
    bool copyToHost(CUDAManager &cudaMgr);
    bool copyPositionsToDevice(CUDAManager &cudaMgr);
    bool copyVelocitiesToDevice(CUDAManager &cudaMgr);
    bool copyMassesToDevice(CUDAManager &cudaMgr);
    bool copyForcesToDevice(CUDAManager &cudaMgr);
    bool copyPotentialsToDevice(CUDAManager &cudaMgr);

    // Memory management
    bool allocateDeviceMemory(CUDAManager &cudaMgr);
    bool freeDeviceMemory(CUDAManager &cudaMgr);
    bool isDeviceMemoryAllocated() const { return deviceArrays.isAllocated(); }

    // Utility methods
    void resetForces();
    void resetPotentials();
    double getTotalMass() const;
    void getCenterOfMass(double *com) const;
    void getTotalMomentum(double *momentum) const;
    double getKineticEnergy() const;
    double getPotentialEnergy() const;

    // File I/O
    bool saveToFile(const std::string &filename) const;
    bool loadFromFile(const std::string &filename);

    // Debug and validation
    void printSummary() const;
    bool validateData() const;

private:
    size_t numParticles;

    // Host data storage (Array of Structures)
    std::vector<double> hostPositions;  // [x1,y1,z1, x2,y2,z2, ...]
    std::vector<double> hostVelocities; // [vx1,vy1,vz1, vx2,vy2,vz2, ...]
    std::vector<double> hostMasses;     // [m1, m2, m3, ...]
    std::vector<double> hostForces;     // [fx1,fy1,fz1, fx2,fy2,fz2, ...]
    std::vector<double> hostPotentials; // [p1, p2, p3, ...]

    // Device data storage (Structure of Arrays)
    ParticleArrays deviceArrays;

    // Helper methods
    void initializeHostData();
    bool allocateHostMemory();
    void generateRandomPositions(double boxSize);
    void generateRandomVelocities(double maxVelocity = 1.0);
    void generateUniformMasses(double mass = 1.0);
    bool validateArraySize(const std::vector<double> &array, size_t expectedSize) const;
    void calculateCenterOfMass(double *com) const;
    void calculateTotalMomentum(double *momentum) const;
};

#endif // PARTICLE_SYSTEM_H