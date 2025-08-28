#ifndef MESH_H
#define MESH_H

#include "cuda_manager.h"
#include <vector>
#include <memory>
#include <cufft.h>

// Mesh grid class for P3M particle-mesh operations
class Mesh
{
public:
    // Constructor and destructor
    Mesh(size_t gridSize, double gridSpacing);
    ~Mesh();

    // Initialization
    bool initialize(CUDAManager &cudaMgr);

    // Grid properties
    size_t getGridSize() const { return gridSize; }
    double getGridSpacing() const { return gridSpacing; }
    size_t getTotalGridPoints() const { return gridSize * gridSize * gridSize; }

    // Particle to mesh operations
    bool assignParticlesToMesh(const std::vector<double> &positions,
                               const std::vector<double> &masses,
                               size_t numParticles);

    // Poisson equation solving
    bool solvePoissonFFT();

    // Force calculation and interpolation
    bool calculateForcesFromPotential();
    bool interpolateForcesToParticles(const std::vector<double> &positions,
                                      std::vector<double> &forces,
                                      size_t numParticles);

    // Combined P3M operation
    bool computePMForces(const std::vector<double> &positions,
                         const std::vector<double> &masses,
                         std::vector<double> &forces,
                         size_t numParticles);

    // Data access (host)
    const std::vector<double> &getHostDensity() const { return hostDensity; }
    const std::vector<double> &getHostPotential() const { return hostPotential; }
    const std::vector<double> &getHostForceX() const { return hostForceX; }
    const std::vector<double> &getHostForceY() const { return hostForceY; }
    const std::vector<double> &getHostForceZ() const { return hostForceZ; }

    // Data access (device)
    double *getDeviceDensity() const { return deviceDensity; }
    double *getDevicePotential() const { return devicePotential; }
    cufftDoubleComplex *getDeviceDensityFFT() const { return deviceDensityFFT; }
    cufftDoubleComplex *getDevicePotentialFFT() const { return devicePotentialFFT; }

    // Memory management
    bool allocateDeviceMemory(CUDAManager &cudaMgr);
    bool freeDeviceMemory(CUDAManager &cudaMgr);
    bool isDeviceMemoryAllocated() const;

    // Data synchronization
    bool copyDensityToDevice(CUDAManager &cudaMgr);
    bool copyPotentialToHost(CUDAManager &cudaMgr);
    bool copyForcesToHost(CUDAManager &cudaMgr);

    // Utility functions
    void resetDensity();
    void resetPotential();
    void resetForces();
    double getTotalMass() const;
    void printGridInfo() const;

    // Green function for Poisson equation in Fourier space
    static double greenFunction(double kx, double ky, double kz, double gridSpacing);

    // Public helper methods for testing
    size_t getGridIndex(size_t i, size_t j, size_t k) const;
    void getGridCoordinates(size_t index, size_t &i, size_t &j, size_t &k) const;
    double cicWeight(double dx) const;

private:
    size_t gridSize;      // Grid size in each dimension (NxNxN)
    double gridSpacing;   // Spacing between grid points (h)
    double gridOrigin[3]; // Grid origin (usually at 0,0,0)

    // Host data
    std::vector<double> hostDensity;   // Mass density ρ(x,y,z)
    std::vector<double> hostPotential; // Gravitational potential Φ(x,y,z)
    std::vector<double> hostForceX;    // Force components F_x(x,y,z)
    std::vector<double> hostForceY;    // Force components F_y(x,y,z)
    std::vector<double> hostForceZ;    // Force components F_z(x,y,z)

    // Device data
    double *deviceDensity;                  // Device density array
    double *devicePotential;                // Device potential array
    cufftDoubleComplex *deviceDensityFFT;   // Device density in Fourier space
    cufftDoubleComplex *devicePotentialFFT; // Device potential in Fourier space

    // cuFFT plans
    cufftHandle forwardPlan; // Forward FFT: real → complex
    cufftHandle inversePlan; // Inverse FFT: complex → real

    // Internal helper methods
    bool createFFTPlans(CUDAManager &cudaMgr);
    bool destroyFFTPlans();
    void setupGreenFunction(cufftDoubleComplex *greenFFT, CUDAManager &cudaMgr);
    void applyCICPAssignment(const std::vector<double> &positions,
                             const std::vector<double> &masses,
                             size_t numParticles);
    void applyCICInterpolation(const std::vector<double> &positions,
                               std::vector<double> &forces,
                               size_t numParticles);

    // CIC (Cloud-in-Cell) interpolation weights
    void cicInterpolate(double px, double py, double pz,
                        double value, std::vector<double> &grid) const;
    double cicInterpolateAt(double px, double py, double pz,
                            const std::vector<double> &grid) const;
};

#endif // MESH_H