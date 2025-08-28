#include "force_calculator.h"
#include "particle_system.h"
#include "mesh.h"
#include "cuda_manager.h"
#include "configuration.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

// Benchmarking configuration
struct BenchmarkConfig
{
    std::vector<size_t> particleCounts = {10, 100, 1000, 10000, 50000};
    std::vector<double> alphaValues = {0.1, 0.2, 0.35, 0.5, 0.7, 1.0};
    std::vector<double> gridSizes = {16, 32, 64}; // Relative to particle count
    double cutoffRadius = 2.5;
    int numTrials = 1; // Number of trials for averaging
};

// Benchmark result structure
struct BenchmarkResult
{
    size_t numParticles;
    double alpha;
    size_t gridSize;
    std::string method;
    double executionTime; // milliseconds
    double maxError;
    double rmsError;
    double potentialEnergy;
    double kineticEnergy;
    size_t memoryUsage; // bytes
};

// Performance measurement class
class PerformanceTimer
{
private:
    std::chrono::high_resolution_clock::time_point startTime;

public:
    void start()
    {
        startTime = std::chrono::high_resolution_clock::now();
    }

    double stop()
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        return duration.count() / 1000.0; // Convert to milliseconds
    }
};

// CSV writer class
class CSVWriter
{
private:
    std::ofstream file;
    bool isFirstRow;

public:
    CSVWriter(const std::string &filename) : isFirstRow(true)
    {
        file.open(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open CSV file: " + filename);
        }
    }

    ~CSVWriter()
    {
        if (file.is_open())
        {
            file.close();
        }
    }

    void writeHeader(const std::vector<std::string> &headers)
    {
        for (size_t i = 0; i < headers.size(); ++i)
        {
            file << headers[i];
            if (i < headers.size() - 1)
                file << ",";
        }
        file << "\n";
    }

    void writeRow(const std::vector<std::string> &values)
    {
        for (size_t i = 0; i < values.size(); ++i)
        {
            file << values[i];
            if (i < values.size() - 1)
                file << ",";
        }
        file << "\n";
    }

    void writeResult(const BenchmarkResult &result)
    {
        if (isFirstRow)
        {
            writeHeader({"numParticles", "alpha", "gridSize", "method",
                         "executionTime", "maxError", "rmsError",
                         "potentialEnergy", "kineticEnergy", "memoryUsage"});
            isFirstRow = false;
        }

        writeRow({std::to_string(result.numParticles),
                  std::to_string(result.alpha),
                  std::to_string(result.gridSize),
                  result.method,
                  std::to_string(result.executionTime),
                  std::to_string(result.maxError),
                  std::to_string(result.rmsError),
                  std::to_string(result.potentialEnergy),
                  std::to_string(result.kineticEnergy),
                  std::to_string(result.memoryUsage)});
    }
};

// Benchmark runner class
class P3MBenchmark
{
private:
    CUDAManager *cudaMgr;
    BenchmarkConfig config;

    // Create test particle system
    ParticleSystem *createTestSystem(size_t numParticles)
    {
        ParticleSystem *particles = new ParticleSystem(numParticles);

        // Create a realistic distribution (not too clustered)
        particles->initializeRandom(*cudaMgr, 5.0); // Box size 5.0

        // Add some bulk motion to make it more realistic
        std::vector<double> velocities(numParticles * 3, 0.0);
        for (size_t i = 0; i < numParticles; ++i)
        {
            velocities[i * 3] = 0.1 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
            velocities[i * 3 + 1] = 0.1 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
            velocities[i * 3 + 2] = 0.1 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        }
        particles->setVelocities(velocities);

        return particles;
    }

    // Calculate memory usage estimate
    size_t estimateMemoryUsage(size_t numParticles, size_t gridSize)
    {
        // Estimate based on data structures used
        size_t particleMemory = numParticles * (3 * sizeof(double) * 4);         // positions, velocities, forces, masses
        size_t gridMemory = gridSize * gridSize * gridSize * sizeof(double) * 4; // density, potential, forces
        return particleMemory + gridMemory;
    }

    // Run single benchmark trial
    BenchmarkResult runTrial(size_t numParticles, double alpha, size_t gridSize,
                             ForceMethod method, bool measureAccuracy = true)
    {
        BenchmarkResult result;
        result.numParticles = numParticles;
        result.alpha = alpha;
        result.gridSize = gridSize;
        result.memoryUsage = estimateMemoryUsage(numParticles, gridSize);

        // Set method name
        switch (method)
        {
        case ForceMethod::DIRECT_SUMMATION:
            result.method = "Direct";
            break;
        case ForceMethod::P3M_FULL:
            result.method = "P3M";
            break;
        default:
            result.method = "Unknown";
            break;
        }

        // Create test system
        ParticleSystem *particles = createTestSystem(numParticles);
        Mesh *mesh = new Mesh(gridSize, 10.0 / gridSize); // Adaptive grid spacing

        // Initialize mesh only if CUDA is available and we're doing P3M
        bool meshInitialized = false;
        if (cudaMgr->isInitialized() && method == ForceMethod::P3M_FULL)
        {
            meshInitialized = mesh->initialize(*cudaMgr);
            if (!meshInitialized)
            {
                std::cerr << "Warning: Mesh initialization failed, skipping P3M test" << std::endl;
            }
        }

        ForceCalculator *forceCalc = new ForceCalculator(particles, mesh, cudaMgr,
                                                         config.cutoffRadius, alpha);

        PerformanceTimer timer;

        try
        {
            // Skip P3M if mesh not initialized
            if (method == ForceMethod::P3M_FULL && !meshInitialized)
            {
                std::cerr << "Skipping P3M test due to mesh initialization failure" << std::endl;
                result.executionTime = -1.0;
            }
            else
            {
                // Warm-up run
                forceCalc->calculateForces(method);

                // Timed run
                timer.start();
                bool success = forceCalc->calculateForces(method);
                result.executionTime = timer.stop();

                if (!success)
                {
                    std::cerr << "Force calculation failed for " << result.method
                              << " with " << numParticles << " particles" << std::endl;
                    result.executionTime = -1.0;
                }
            }

            // Get energy values
            result.potentialEnergy = forceCalc->getPotentialEnergy();
            result.kineticEnergy = forceCalc->getKineticEnergy();

            // Accuracy comparison (only for P3M vs Direct)
            if (measureAccuracy && method == ForceMethod::P3M_FULL && numParticles <= 1000)
            {
                double maxError, rmsError;
                if (forceCalc->compareWithDirectSummation(maxError, rmsError, std::min(numParticles, size_t(100))))
                {
                    result.maxError = maxError;
                    result.rmsError = rmsError;
                }
                else
                {
                    result.maxError = -1.0;
                    result.rmsError = -1.0;
                }
            }
            else
            {
                result.maxError = 0.0; // Not measured
                result.rmsError = 0.0;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Benchmark trial failed: " << e.what() << std::endl;
            result.executionTime = -1.0;
            result.maxError = -1.0;
            result.rmsError = -1.0;
        }

        // Cleanup
        delete forceCalc;
        delete mesh;
        delete particles;

        return result;
    }

public:
    P3MBenchmark() : cudaMgr(new CUDAManager())
    {
        // Try to initialize CUDA
        try
        {
            cudaMgr->initializeDevice(0);
        }
        catch (const CUDAException &)
        {
            std::cout << "CUDA not available, using CPU-only benchmarking" << std::endl;
        }
    }

    ~P3MBenchmark()
    {
        delete cudaMgr;
    }

    // Run performance benchmark
    void runPerformanceBenchmark(const std::string &outputFile)
    {
        std::cout << "Running performance benchmark..." << std::endl;

        CSVWriter writer(outputFile);

        for (size_t numParticles : config.particleCounts)
        {
            std::cout << "Testing " << numParticles << " particles..." << std::endl;

            // Test Direct Summation
            if (numParticles <= 1000)
            { // Only for smaller systems due to O(N²)
                std::vector<double> times;
                for (int trial = 0; trial < config.numTrials; ++trial)
                {
                    auto result = runTrial(numParticles, 0.35, 32, ForceMethod::DIRECT_SUMMATION, false);
                    if (result.executionTime > 0)
                    {
                        times.push_back(result.executionTime);
                    }
                }

                if (!times.empty())
                {
                    double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                    BenchmarkResult avgResult = runTrial(numParticles, 0.35, 32, ForceMethod::DIRECT_SUMMATION, false);
                    avgResult.executionTime = avgTime;
                    writer.writeResult(avgResult);
                }
            }

            // Test P3M with different grid sizes (only if CUDA is available)
            if (cudaMgr->isInitialized())
            {
                for (size_t gridSize : config.gridSizes)
                {
                    if (gridSize * gridSize * gridSize > 1000000)
                        continue; // Skip too large grids

                    std::vector<double> times;
                    for (int trial = 0; trial < config.numTrials; ++trial)
                    {
                        auto result = runTrial(numParticles, 0.35, gridSize, ForceMethod::P3M_FULL, false);
                        if (result.executionTime > 0)
                        {
                            times.push_back(result.executionTime);
                        }
                    }

                    if (!times.empty())
                    {
                        double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                        BenchmarkResult avgResult = runTrial(numParticles, 0.35, gridSize, ForceMethod::P3M_FULL, false);
                        avgResult.executionTime = avgTime;
                        writer.writeResult(avgResult);
                    }
                }
            }
            else
            {
                std::cout << "  Skipping P3M tests (CUDA not available)" << std::endl;
            }
        }

        std::cout << "Performance benchmark completed. Results saved to " << outputFile << std::endl;
    }

    // Run precision benchmark
    void runPrecisionBenchmark(const std::string &outputFile)
    {
        std::cout << "Running precision benchmark..." << std::endl;

        CSVWriter writer(outputFile);

        // Use a moderate particle count for precision testing
        size_t testParticles = 500;

        if (cudaMgr->isInitialized())
        {
            for (double alpha : config.alphaValues)
            {
                std::cout << "Testing alpha = " << alpha << "..." << std::endl;

                // Test with different grid sizes
                for (size_t gridSize : {32, 64})
                {
                    std::vector<BenchmarkResult> results;
                    for (int trial = 0; trial < config.numTrials; ++trial)
                    {
                        auto result = runTrial(testParticles, alpha, gridSize, ForceMethod::P3M_FULL, true);
                        if (result.executionTime > 0)
                        {
                            results.push_back(result);
                        }
                    }

                    if (!results.empty())
                    {
                        // Average the results
                        BenchmarkResult avgResult = results[0];
                        for (size_t i = 1; i < results.size(); ++i)
                        {
                            avgResult.executionTime += results[i].executionTime;
                            avgResult.maxError += results[i].maxError;
                            avgResult.rmsError += results[i].rmsError;
                        }
                        avgResult.executionTime /= results.size();
                        avgResult.maxError /= results.size();
                        avgResult.rmsError /= results.size();

                        writer.writeResult(avgResult);
                    }
                }
            }
        }
        else
        {
            std::cout << "Skipping precision benchmark (CUDA not available)" << std::endl;
        }

        std::cout << "Precision benchmark completed. Results saved to " << outputFile << std::endl;
    }

    // Run scaling benchmark
    void runScalingBenchmark(const std::string &outputFile)
    {
        std::cout << "Running scaling benchmark..." << std::endl;

        CSVWriter writer(outputFile);

        // Test scaling for both methods
        std::vector<size_t> scalingParticles = {100, 200, 500, 1000, 2000, 5000};

        // Direct summation scaling (only small systems)
        for (size_t numParticles : scalingParticles)
        {
            if (numParticles > 1000)
                break; // Skip large systems for direct summation

            auto result = runTrial(numParticles, 0.35, 32, ForceMethod::DIRECT_SUMMATION, false);
            if (result.executionTime > 0)
            {
                writer.writeResult(result);
            }
        }

        // P3M scaling (only if CUDA is available)
        if (cudaMgr->isInitialized())
        {
            for (size_t numParticles : scalingParticles)
            {
                size_t gridSize = std::max(size_t(16), size_t(std::cbrt(numParticles) * 1.5));

                auto result = runTrial(numParticles, 0.35, gridSize, ForceMethod::P3M_FULL, false);
                if (result.executionTime > 0)
                {
                    writer.writeResult(result);
                }
            }
        }
        else
        {
            std::cout << "Skipping P3M scaling tests (CUDA not available)" << std::endl;
        }

        std::cout << "Scaling benchmark completed. Results saved to " << outputFile << std::endl;
    }
};

// Main function
int main(int argc, char *argv[])
{
    std::cout << "=== P3M Benchmark Suite ===" << std::endl;
    std::cout << "Comprehensive benchmarking of P3M vs Direct Summation" << std::endl;
    std::cout << std::endl;

    try
    {
        P3MBenchmark benchmark;

        // Run performance benchmark
        benchmark.runPerformanceBenchmark("performance.csv");

        // Run precision benchmark
        benchmark.runPrecisionBenchmark("precision.csv");

        // Run scaling benchmark
        benchmark.runScalingBenchmark("scaling.csv");

        std::cout << std::endl;
        std::cout << "All benchmarks completed successfully!" << std::endl;
        std::cout << "Generated files:" << std::endl;
        std::cout << "  - performance.csv" << std::endl;
        std::cout << "  - precision.csv" << std::endl;
        std::cout << "  - scaling.csv" << std::endl;
        std::cout << std::endl;
        std::cout << "Use the Python visualization script to analyze results." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}