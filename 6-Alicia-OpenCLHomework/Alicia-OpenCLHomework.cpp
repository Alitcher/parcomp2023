#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

const int M = 512;
const int N = 512;
const int K = 512;

void initOpenCL(cl::Context& context, cl::Device& device, cl::Program& program) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        exit(1);
    }

    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
    context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    if (devices.empty()) {
        std::cerr << "No OpenCL devices found." << std::endl;
        exit(1);
    }

    device = devices[0];

    std::ifstream sourceFile("matrix_multiply.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources sources(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
    program = cl::Program(context, sources);

    try {
        program.build(devices);
    }
    catch (const cl::Error& e) {
        std::cerr << "Build error: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }
}

class MatrixOperations {
public:
    MatrixOperations(int rows, int cols) : rows_(rows), cols_(cols) {}

    void generateRandomMatrix(std::vector<float>& matrix) {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

        matrix.resize(rows_ * cols_);
        for (int i = 0; i < rows_ * cols_; i++) {
            matrix[i] = distribution(generator);
        }
    }

    void matrixMultiply(cl::Context& context, cl::Device& device, cl::Program& program,
        const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {

        // Create OpenCL buffers
        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.size(), const_cast<float*>(A.data()));
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * B.size(), const_cast<float*>(B.data()));
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * C.size());

        // Create kernel
        cl::Kernel kernel(program, "matrix_multiply");

        // Set kernel arguments
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, rows_);
        kernel.setArg(4, cols_);
        kernel.setArg(5, rows_); // K should be rows_ for a square matrix

        // Create command queue
        cl::CommandQueue queue(context, device);

        // Execute the kernel
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows_, cols_), cl::NullRange, NULL, &event);
        event.wait();

        // Read the result back to host
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * C.size(), C.data());
    }

    void matrixMultiplySequential(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
        for (int m = 0; m < rows_; m++) {
            for (int n = 0; n < cols_; n++) {
                float acc = 0.0f;
                for (int k = 0; k < rows_; k++) {
                    acc += A[k * rows_ + m] * B[n * rows_ + k];
                }
                C[n * rows_ + m] = acc;
            }
        }
    }

private:
    int rows_;
    int cols_;
};

void Task1(cl::Context& context, cl::Device& device, cl::Program& program) {
    int M = 512;
    int N = 512;
    int K = 512;

    MatrixOperations matrixOps(M, N);

    initOpenCL(context, device, program);

    std::vector<float> A, B, C;
    matrixOps.generateRandomMatrix(A);
    matrixOps.generateRandomMatrix(B);
    C.resize(M * N);

    // Measure execution time for OpenCL matrix multiplication
    auto startTime = std::chrono::high_resolution_clock::now();
    matrixOps.matrixMultiply(context, device, program, A, B, C);
    auto endTime = std::chrono::high_resolution_clock::now();
    double executionTimeOpenCL = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    std::cout << "OpenCL Execution time: " << executionTimeOpenCL << " ms" << std::endl;

    // Measure execution time for sequential matrix multiplication
    std::vector<float> D, E, F;
    matrixOps.generateRandomMatrix(D);
    matrixOps.generateRandomMatrix(E);
    F.resize(M * N);

    startTime = std::chrono::high_resolution_clock::now();
    matrixOps.matrixMultiplySequential(D, E, F);
    endTime = std::chrono::high_resolution_clock::now();
    double executionTimeSequential = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Sequential Execution time: " << executionTimeSequential << " ms" << std::endl;
}

void Task2() 
{

}

int main() {
    cl::Context context;
    cl::Device device;
    cl::Program program;
    initOpenCL(context, device, program);


    /*Task 6.1*/
    Task1(context, device, program);

    return 0;
}
