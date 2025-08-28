#ifndef FORCE_CALCULATOR_H
#define FORCE_CALCULATOR_H

#include "particle_system.h"
#include "mesh.h"
#include "cuda_manager.h"
#include <vector>
#include <memory>

// Force calculation method enumeration
enum class ForceMethod
{
    DIRECT_SUMMATION,  // O(N²) direct calculation
    PARTICLE_MESH,     // P3M with mesh only
    PARTICLE_PARTICLE, // Direct PP only
    P3M_FULL           // Complete P3M (PP + PM)
};

// ForceCalculator class implementing P3M algorithm
class ForceCalculator
{
public:
    // Constructor and destructor
    ForceCalculator(ParticleSystem *particles, Mesh *mesh, CUDAManager *cudaMgr,
                    double cutoffRadius, double ewaldAlpha = 0.5);
    ~ForceCalculator();

    // Main force calculation methods
    bool calculateForces(ForceMethod method = ForceMethod::P3M_FULL);

    // Individual P3M components
    bool calculatePPForces();     // Particle-Particle forces
    bool calculatePMForces();     // Particle-Mesh forces
    bool calculateDirectForces(); // Direct summation (for validation)

    // Force combination and application
    bool combineForces();
    bool applyForcesToParticles();

    // Parameter access and modification
    void setCutoffRadius(double cutoff);
    void setEwaldAlpha(double alpha);
    void setSofteningParameter(double softening);
    double getCutoffRadius() const { return cutoffRadius; }
    double getEwaldAlpha() const { return ewaldAlpha; }

    // Energy calculations
    double getPotentialEnergy() const;
    double getKineticEnergy() const;

    // Validation and comparison methods
    bool compareWithDirectSummation(double &maxError, double &rmsError,
                                    size_t numTestParticles = 100);

    // Performance and statistics
    void printPerformanceStats() const;
    size_t getNumForceEvaluations() const { return numForceEvaluations; }

    // Force access for testing
    const std::vector<double> &getPPForces() const { return ppForces; }
    const std::vector<double> &getPMForces() const { return pmForces; }
    const std::vector<double> &getTotalForces() const { return totalForces; }

private:
    // Core components
    ParticleSystem *particles;
    Mesh *mesh;
    CUDAManager *cudaMgr;

    // P3M parameters
    double cutoffRadius;       // Cutoff for short-range forces
    double ewaldAlpha;         // Ewald splitting parameter
    double softeningParameter; // Force softening parameter

    // Force storage (3 components per particle: Fx, Fy, Fz)
    std::vector<double> ppForces;    // Particle-particle forces
    std::vector<double> pmForces;    // Particle-mesh forces
    std::vector<double> totalForces; // Combined forces

    // Energy and statistics
    double potentialEnergy;
    double kineticEnergy;
    size_t numForceEvaluations;

    // Internal calculation methods
    bool calculatePPForcesCPU();     // CPU implementation of PP forces
    bool calculatePPForcesGPU();     // GPU implementation of PP forces
    bool calculateDirectForcesCPU(); // CPU direct summation

    // Ewald splitting functions
    double erfcEwald(double r, double alpha) const;
    double ewaldShortRangePotential(double r, double alpha) const;
    double ewaldShortRangeForce(double r, double alpha) const;

    // Force computation kernels (would be in .cu file)
    void computePPForcesKernel(const double *positions, const double *masses,
                               double *forces, size_t numParticles,
                               double cutoff, double alpha, double softening);

    // Utility methods
    double calculateDistance(const double *pos1, const double *pos2) const;
    void calculateForceComponents(const double *pos1, const double *pos2,
                                  double mass1, double mass2,
                                  double *force, double &potential) const;
    bool validateInputs() const;
    void resetForces();
    void initializeForceArrays();

    // Periodic boundary conditions
    void applyPeriodicBoundary(double *pos) const;
    double applyMinimumImage(double dx) const;
};

#endif // FORCE_CALCULATOR_H