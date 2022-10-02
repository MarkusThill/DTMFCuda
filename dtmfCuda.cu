#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>
#include <vector>
#include <cstring>
#include <cstdio>
#include <tuple>
#include <cassert>
#include <map>
#include <filesystem>
#include <cuda_runtime_api.h>

#define DEBUG

/*!
 * Macro, determining the thread index based on block and grid index
 */
#define COMPUTE_THREAD_IDX() int blockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y); \
        int i = blockId * (blockDim.x * blockDim.y * blockDim.z) \
        + threadIdx.x \
        + threadIdx.y * blockDim.x \
        + threadIdx.z * blockDim.x * blockDim.y \

/*!
 * Mostly based on: https://stackoverflow.com/a/32128050
 */
typedef struct WAV_HEADER {
    /* RIFF Chunk Descriptor */
    uint8_t RIFF[4];        // RIFF Header Magic header
    uint32_t ChunkSize;      // RIFF Chunk Size
    uint8_t WAVE[4];        // WAVE Header
    /* "fmt" sub-chunk */
    uint8_t fmt[4];         // FMT header
    uint32_t subchunk1Size;  // Size of the fmt chunk
    uint16_t AudioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw,     257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
    uint16_t NumOfChan;      // Number of channels 1=Mono 2=Stereo
    uint32_t SamplesPerSec;  // Sampling Frequency in Hz
    uint32_t bytesPerSec;    // bytes per second
    uint16_t blockAlign;     // 2=16-bit mono, 4=16-bit stereo
    uint16_t bitsPerSample;  // Number of bits per sample
    /* "data" sub-chunk */
    uint8_t subchunk2Id[4]; // "data"  string
    uint32_t subchunk2Size;  // Sampled data length
} wav_hdr;

namespace fs = std::filesystem;

// =======================================================================================
// Constants
// =======================================================================================
/*!
 * Filter length used for the bandpass filters. Generally, larger values correspond to a "sharper" frequency response
 * and result in a "clearer" filtering of the signal. For this example, filter lengths larger than 40 appear to work
 * reliably. In the range [10,100], a filter length of 71 is optimal for the selected filter design method.
 */
static const int FILTER_LENGTH = 71;

/*!
 * Since the DTMF signal is oversampled, we decimate it by a specified factor. This reduces the computational
 * effort. As we do not low-pass filter the signal before decimation, there might be some aliasing effects, which
 * we ignore for this example.
 */
static const int DECIMATION_FACTOR = 8;

/*!
 * At a later stage in the data processing, we have longer sequences containing the frequency values, which were
 * originally present in the signal. Since we do not care about the length of these sequences (a subsequence containing
 * the frequencies 697 Hz and 1209 Hz will represent the symbol "1", no matter if it has a length of 100 or 1000). Hence,
 * we perform several pooling steps (in each step the sequence length is halved), to significantly reduce the final
 * computation tasks.
 */
static const int NUM_POOLING_STEPS = 5;

/*!
 * All frequencies that can be present in a DTMF signal. We will later construct FIR bandpass filters for each frequency.
 */
static const std::vector<float> DTMF_FREQUENCIES{697, 770, 852, 941, 1209, 1336, 1477, 1633};

/*!
 * Verbosity: 0-> Only print errors, 1->Only print results, 2-> print everything
 */
static const int VERBOSE = 1;

/*!
 * Mapping from frequencies to symbol (each symbol is characterized by 2 frequencies)
 * DTMF keypad frequencies: https://en.wikipedia.org/wiki/Dual-tone_multi-frequency_signaling#Keypad

                  1209 1336 1477 1633
            697    1    2    3    A
            770    4    5    6    B
            852    7    8    9    C
            941    *    0    #    D
 */
static const std::map<int, std::map<int, std::string>> freqToSymbolMap{
        {697, {{1209, "1"}, {1336, "2"}, {1477, "3"}, {1633, "A"}}},
        {770, {{1209, "4"}, {1336, "5"}, {1477, "6"}, {1633, "B"}}},
        {852, {{1209, "7"}, {1336, "8"}, {1477, "9"}, {1633, "C"}}},
        {941, {{1209, "*"}, {1336, "0"}, {1477, "#"}, {1633, "D"}}}};


// =======================================================================================
// kernel Functions
// =======================================================================================

/*!
 * Divides all elements of an array by a constant value and stores the results in a new array
 * @param dIn Input array with integer elements
 * @param dOut Output array containing the elements of `dIn` divided by the constant `c` as floats
 * @param c Scalar divisor
 * @param numElements Size of `dIn` and `dOut`
 */
__global__ void divIntByFloat(const int *const dIn, float *const dOut, const float c, const int numElements) {
    COMPUTE_THREAD_IDX();
    if (i < numElements) {
        dOut[i] = float(dIn[i]) / c;
    }
}

/*!
 * Divides all elements of an array by a constant value and stores the results in a the same array
 * @param dInOut Array, containing `numElements` float values
 * @param c Scalar divisor
 * @param numElements Size of dInOut
 */
__global__ void divFloat(float *const dInOut, const float c, const int numElements) {
    COMPUTE_THREAD_IDX();
    if (i < numElements) {
        dInOut[i] = dInOut[i] / c;
    }
}

/*!
 * Multiplies all elements of an array by a constant value and stores the results in a the same array
 * @param dInOut Array, containing `numElements` integer values
 * @param c Scalar integer multiplicand
 * @param numElements Size of dInOut
 */
__global__ void multInt(int *const dInOut, const int c, const int numElements) {
    COMPUTE_THREAD_IDX();
    if (i < numElements) {
        dInOut[i] = dInOut[i] * c;
    }
}

/*!
 * Squares an input array and writes the results into an output array
 * @param dIn Input array
 * @param dOut Output array
 * @param numElements Size of input/output array
 */
__global__ void squared(const float *const dIn, float *const dOut, const int numElements) {
    COMPUTE_THREAD_IDX();
    if (i < numElements) {
        dOut[i] = dIn[i] * dIn[i];
    }
}

/*!
 * Down-samples an input array by a given rate.
 * @param dIn Input array
 * @param dOut Output array
 * @param decimateFactor Decimation factor
 * @param numElements Size of the input array `dIn`
 */
__global__ void decimate(const float *const dIn, float *const dOut, const int decimateFactor, const int numElements) {
    COMPUTE_THREAD_IDX();
    if (i * decimateFactor < numElements) {
        dOut[i] = float(dIn[i * decimateFactor]);
    }
}

/*!
 * Kernel that can be used to compute the sum over an array. Note that the kernel has to be called multiple times
 * (depending on the size of the array) in order to obtain the overall sum. The idea of this approach is to iteratively
 * add pairs of elements in the array, store the result of each addition in the left pair element and then increase the
 * distance between the pair elements.
 * After the required number of calls, the  value of the sum is located at the left-most element of the array.
 * Note that the array is modified in-place during the computation.
 * The runtime complexity of this approach is logarithmic. In each iteration, this kernel is called
 * with the `factor` argument doubled (starting with 1), until `factor` > `numElements`.
 * @param dInOut Array, for which we want to compute the sum over all elements
 * @param factor Factor, which is doubled in each iteration (starting with 1)
 * @param numElements Total size of the array
 */
__global__ void sum(float *const dInOut, const int factor, const int numElements) {
    COMPUTE_THREAD_IDX();
    if ((i + 1) * factor < numElements) {
        dInOut[i * factor] = dInOut[i * factor] + dInOut[(i + 1) * factor];
    }
}


/*!
 * Perform max pooling on an input array. The function looks at adjacent pairs of values and selects the maximum in
 * each pair and writes it to the output array.
 * At a later stage in the data processing, we have longer sequences containing the frequency values, which were
 * originally present in the signal. Since we do not care about the length of these sequences (a subsequence containing
 * the frequencies 697 Hz and 1209 Hz will represent the symbol "1", no matter if it has a length of 100 or 1000). Hence,
 * we perform several pooling steps (in each step the sequence length is halved), to significantly reduce the final
 * computation tasks.
 * @param dIn Input array
 * @param dOut Output array where the pooled values are stored
 * @param pooledSize Size of the output array `dOut`. Ensure that the input array `dIn` has twice the size of `dOut`.
 */
__global__ void maxPool(const int *const dIn, int *const dOut, const int pooledSize) {
    COMPUTE_THREAD_IDX();
    if (i < pooledSize) {
        dOut[i] = max(dIn[2 * i], dIn[2 * i + 1]);
    }
}

/*!
 * Convolve an input sequence (`dIn`) with the impulse response of a FIR filter (`dhh`). The convolution operation
 * can efficiently parallelized on the GPU.
 * @param dIn Input sequence
 * @param dOut Output, where the result of the convolution operation is written
 * @param dhh Impulse response (filter weights) of the FIR filter
 * @param L Length of the impulse response
 * @param numElements Length of the input signal
 */
__global__ void convolve(const float *const dIn,
                         float *const dOut,
                         const float *const dhh,
                         const int L,
                         const int numElements) {
    COMPUTE_THREAD_IDX();
    if (i < numElements - (L - 1)) {
        dOut[i] = 0.0;
        // iterate through filter elements
        for (int j = 0; j < L; j++) {
            dOut[i] += (dhh[j] * dIn[i + L - 1 - j]);
        }
    }
}

/*!
 * Check for each element of an input array `dIn`, if its value exceeds a given scalar value (`cmp`) and write a binary
 * flag (0,1) to an output array `dOut`
 * @param dIn Input array
 * @param dOut Output array
 * @param cmp Comparison threshold. If an element in `dIn` is larger than this value, the corresponding element in
 * `dOut` is set.
 * @param numElements Size of `dIn` and `dOut`
 */
__global__ void largerThan(const float *const dIn, int *const dOut, const float *cmp, const int numElements) {
    COMPUTE_THREAD_IDX();
    if (i < numElements) {
        dOut[i] = (dIn[i] > *cmp);
    }
}

/*!
 * Finds the largest 2 values among a set of `numDimsIn` sequences of equal length along the time axis. Hence a sequence
 * with the shape [`numElements` x `numDimsIn`] is reduced to a sequence of shape [`numElements` x 2]. This kernel is
 * used to identify the two frequencies, which characterize a symbol in the DTMF signal.
 * @param dIn Input array, containing `numDimsIn` subsequences of length `numElements`. The array holds all
 * subsequences in the form |subsequence_1|subsequence_2|...|subsequence_{numDimsIn}|.
 * @param dOutOutput array, containing 2 subsequences of length `numElements` in the form |subsequence_1|subsequence_2|.
 * @param numDimsIn Dimensionality of the input sequence (in this application it will be 8, since there are 8 possible
 * frequencies that might appear in a signal.
 * @param numElements Length of the input sub-sequences
 */
__global__ void top2(const int *const dIn, int *const dOut, const int numDimsIn, const int numElements) {
    COMPUTE_THREAD_IDX();
    if (i < numElements) {
        int t2 = 0;
        for (int j = 0; j < numDimsIn; j++) {
            int idx_in = i + j * numElements;
            if (dIn[idx_in] > 0) {
                int idx_out = i + numElements * t2++;
                dOut[idx_out] = dIn[idx_in];
            }
        }
        if (t2 != 2) {
            // There are cases where only one frequency > 0 is found. Also, there might be cases, where more than 2
            // frequencies where found. Ignore these cases.
            dOut[i + numElements] = dOut[i] = 0;
        }
    }
}


// =======================================================================================
// Host Functions
// =======================================================================================
/*!
 * Determines the size of a file in bytes.
 * @param inFile File, for which the size is to be determined
 * @return The size of the file in bytes
 */
__host__ size_t getFileSize(FILE *inFile) {
    size_t fileSize = 0;
    fseek(inFile, 0, SEEK_END);

    fileSize = ftell(inFile);

    fseek(inFile, 0, SEEK_SET);
    return fileSize;
}

/*!
 * A simple function that determines the dimensions of the grid and its blocks based on the required number of threads.
 * Essentially, the function simply iterates through all grid and block axis and multiplies their size by a factor
 * of 2 until the desired total size is reached.
 * @param totalSize The total number of required threads.
 * @return the block and grid layout (in this order)
 */
__host__ std::tuple<dim3, dim3> getBlockGridSize(size_t totalSize) {
    int dims[] = {1, 1, 1, 2, 2, 2};
    // prepare the kernel dims
    int curSize = 8, j = 0;
    while (curSize < totalSize) {
        curSize <<= 1;
        dims[j++ % 6] <<= 1;
    }
    dim3 block(dims[0], dims[1], dims[2]);
    dim3 grid(dims[3], dims[4], dims[5]);

    auto result = std::make_tuple(block, grid);
    return result;
}

/*!
 * Checks the result of a CUDA operation. Convenience function for checking CUDA runtime API results.
 * It can be wrapped around any runtime API call. No-op in release builds.
 * @param result
 * @return Returns the function argument
 */
__host__ inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

/*!
 * Mostly based on: https://stackoverflow.com/a/32128050
 * @param filePath Path to WAV file to be opened
 * @return returns the header of the WAV file and the data in a std::vector. The data is assumed to be in integer
 * format.
 */
__host__ std::tuple<wav_hdr, std::vector<int>> readWAVFile(const std::string &filePath) {
    wav_hdr wavHeader;
    int headerSize = sizeof(wav_hdr);
    size_t filelength = 0;

    FILE *wavFile = fopen(filePath.c_str(), "r");
    if (wavFile == nullptr) {
        fprintf(stderr, "Unable to open wave file: %s\n", filePath.c_str());
        exit(EXIT_FAILURE);
    }

    std::vector<int> data;

    //Read the header
    size_t bytesRead = fread(&wavHeader, 1, headerSize, wavFile);
    if (VERBOSE > 1) std::cout << "Header Read " << bytesRead << " bytes." << std::endl;
    if (bytesRead > 0) {
        //Read the data
        uint16_t bytesPerSample = wavHeader.bitsPerSample / 8;      //Number     of bytes per sample
        uint64_t numSamples = wavHeader.ChunkSize / bytesPerSample; //How many samples are in the wav file?
        static const uint16_t BUFFER_SIZE = 4096;
        auto *buffer = new int[BUFFER_SIZE];
        size_t elemsRead;
        while ((elemsRead = fread(buffer, sizeof(int), BUFFER_SIZE, wavFile)) > 0) {
            data.insert(data.end(), buffer, buffer + elemsRead);
        }
        delete[] buffer;
        buffer = nullptr;
        filelength = getFileSize(wavFile);
        if (VERBOSE > 1) {
            std::cout << "File is                    : " << filelength << " bytes." << std::endl;
            std::cout << "RIFF header                : " << wavHeader.RIFF[0] << wavHeader.RIFF[1] << wavHeader.RIFF[2]
                      << wavHeader.RIFF[3] << std::endl;
            std::cout << "WAVE header                : " << wavHeader.WAVE[0] << wavHeader.WAVE[1] << wavHeader.WAVE[2]
                      << wavHeader.WAVE[3] << std::endl;
            std::cout << "FMT                        : " << wavHeader.fmt[0] << wavHeader.fmt[1] << wavHeader.fmt[2]
                      << wavHeader.fmt[3] << std::endl;
            std::cout << "Data size                  : " << wavHeader.ChunkSize << std::endl;
            std::cout << "Num samples                : " << numSamples << std::endl;

            // Display the sampling Rate from the header
            std::cout << "Sampling Rate              : " << wavHeader.SamplesPerSec << std::endl;
            std::cout << "Number of bits used        : " << wavHeader.bitsPerSample << std::endl;
            std::cout << "Number of channels         : " << wavHeader.NumOfChan << std::endl;
            std::cout << "Number of bytes per second : " << wavHeader.bytesPerSec << std::endl;
            std::cout << "Data length                : " << wavHeader.subchunk2Size << std::endl;
            std::cout << "Audio Format               : " << wavHeader.AudioFormat << std::endl;
            // Audio format 1=PCM,6=mulaw,7=alaw, 257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM

            std::cout << "Block align                : " << wavHeader.blockAlign << std::endl;
            std::cout << "Size of the fmt chunk      : " << wavHeader.subchunk1Size << std::endl;
            std::cout << "Data string                : " << wavHeader.subchunk2Id[0] << wavHeader.subchunk2Id[1]
                      << wavHeader.subchunk2Id[2] << wavHeader.subchunk2Id[3] << std::endl;

            std::cout << "Size of data vector        : " << data.size() << std::endl;
        }
    }
    fclose(wavFile);
    auto result = std::make_tuple(wavHeader, data);
    return result;
}

/*!
 * Compute the filter coefficients of a simple bandpass filter for a bandpass frequency `f_b` and a sample frequency
 * `f_s`. The generated filters are by no means optimal and only chosen here for reasons of simplicity. The filter
 * weights are scaled in a way that the magnitude of the frequency response at `fb` is exactly one and tends towards
 * zero for larger or smaller frequencies.
 * @param f_b Bandpass frequency
 * @param f_s Sample frequency
 * @param filter_len Length of the desired filter. Generally, a larger value will result in a "sharper" frequency
 * response.
 * @return filter coefficients for the bandpass filter.
 */
__host__ std::vector<float> filterCoeff(const float f_b, const float f_s, const int filter_len) {
    std::vector<float> hh;
    int L = filter_len;
    float w0 = 2.0F * float(M_PI) * f_b / f_s;

    float beta_2_A = 1.0F / 2.0F * sin(w0 * float(L)) / sin(w0);
    float beta_real = L / 2 + beta_2_A * cos(w0 * float(L - 1));
    float beta_imag = -beta_2_A * sin(w0 * float(L - 1));
    float beta = sqrt(beta_real * beta_real + beta_imag * beta_imag);

    for (int k = 0; k < L; k++) {
        hh.push_back(1.0F / beta * cos(w0 * float(k)));
    }
    return hh;
}

/*!
 * Square a set of signals contained in `dSignal`. The number of signals 'n' in `dSignal` is specified by the number of
 * CUDA streams (`streams.size()`). For each signal an independent CUDA stream is used (since previous and subsequent
 * operations might also been performed asynchronously).
 * @param dSignal Contains the signals/subsequences in the form |subsequence_0|subsequence_1|...|subsequence_n|. The
 * overall length of `dSignal` is `streams.size() * dataSize`.
 * @param streams One CUDA stream per subsequence is required
 * @param dataSize The length of an individual subsequence.
 */
__host__ void squareSignals(float *const dSignal,
                            const std::vector<cudaStream_t> &streams,
                            const size_t dataSize) {
    auto [block, grid] = getBlockGridSize(dataSize);
    for (int i = 0; i < streams.size(); i++) {
        auto dos = dSignal + i * dataSize;
        squared<<<grid, block, 0, streams[i]>>>(dos, dos, dataSize);
        checkCuda(cudaGetLastError());
    }
}

/*!
 * Compute the signal power in a sliding window which is moved over the input signals (which have been squared already).
 * @param dSignalSquared Contains the signals/subsequences in the form |subsequence_0|subsequence_1|...|subsequence_n|.
 * The overall length of `dSignal` is `streams.size() * dataSize`. The function slides a window over every
 * subsequence and averages the values within the window to produce one element in the output sequence.
 * @param streams One CUDA stream per subsequence is required
 * @param dataSize Total size of the input array `dSignalSquared`
 * @return Returns
 *      1. A pointer to the weights of the averaging filter in the device memory. Since the kernel calls in this
 *         function are asynchronous and no explicit synchronization is performed this memory has to be released later
 *         when it can be ensured that the memory is no longer required.
 *      2. An array with the same shape as `dSignalSquared`. The elements of the output array are the averaged values
 *         of the sliding window which was moved over the input subsequences
 */
__host__ std::tuple<float *, float *> slidingWinSignalPower(float *const dSignalSquared,
                                                            const std::vector<cudaStream_t> &streams,
                                                            const size_t dataSize) {
    int L = 100;
    float *dhh, *hhh, *dOut;
    checkCuda(cudaMalloc((void **) &dOut, dataSize * streams.size() * sizeof(float)));
    checkCuda(cudaMalloc((void **) &dhh, L * sizeof(float)));
    checkCuda(cudaMallocHost((void **) &hhh, L * sizeof(float)));
    for (int i = 0; i < L; i++) {
        hhh[i] = 1.0F / float(L); // define an averaging filter
    }
    // Create a stream for copying the weights of the averaging filter to the device
    // We could also use the default stream in this case...
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));
    checkCuda(cudaMemcpyAsync(dhh, hhh, L * sizeof(float), cudaMemcpyHostToDevice, stream));
    // We have to wait until the filter weights are on the GPU (should not block too long)...
    cudaStreamSynchronize(stream);

    // Now we can call all convolve operations in parallel
    auto [block, grid] = getBlockGridSize(dataSize);
    for (int i = 0; i < streams.size(); i++) {
        auto dIn_i = dSignalSquared + i * dataSize;
        auto dOut_i = dOut + i * dataSize;
        convolve<<<grid, block, 0, streams[i]>>>(dIn_i, dOut_i, dhh, L, dataSize);
        checkCuda(cudaGetLastError());
    }
    // stream & hhh not required any more, since we can make sure that all the weights have been copied to the device
    cudaStreamDestroy(stream);
    cudaFreeHost(hhh);

    // Since the convolution operations are asynchronous, we cannot delete dhh yet. Hence, return it and delete later
    return {dhh, dOut};
}

/*!
 * Copies the WAV data from the host to the device. For illustration purposes a pinned memory area is created on
 * the host side and the WAV data is first copied to the pinned memory before being transferred to the device.
 * @param data The WAV data
 * @param stream The CUDA stream to be used
 * @return Returns a pointer to the pinned host memory that was created and a pointer to the device memory. Note that
 * this function does not free the memory allocated within. Hence, it has to be ensured that `cudaFreeHost()` and
 * `cudaFree` are called, when the memory is no longer required.
 */
__host__ std::tuple<int *, int *> copyWavDataToDevice(const std::vector<int> &data, const cudaStream_t &stream) {
    // allocate and initialize for data array
    int *hPinnedDataInt;
    int *dDataInt;
    checkCuda(cudaMallocHost((void **) &hPinnedDataInt, data.size() * sizeof(int))); // host pinned
    checkCuda(cudaMalloc((void **) &dDataInt, data.size() * sizeof(int)));           // device

    // Somewhat unnecessary and just done for illustration purposes: fill the pinned host memory
    std::copy(data.begin(), data.end(), hPinnedDataInt);

    // Copy from host to device
    checkCuda(cudaMemcpyAsync(dDataInt, hPinnedDataInt, data.size() * sizeof(int), cudaMemcpyHostToDevice, stream));

    return std::tuple<int *, int *>{hPinnedDataInt, dDataInt};
}

/*!
 * Since the WAV audio data is represented by integer values, we convert the data to float format and
 * then scale the data into the float range [-1.0, 1.0]. Generally, it is not necessary to scale the data. We do it
 * here for convenience reasons.
 *
 * @param dDataInt input data in signed integer format
 * @param stream CUDA stream to use for the conversion of the data on the device
 * @param dataSize Total size of the input array `dDataInt`
 * @param scaleVal Value used to scale the data. All elements in the array are divided by this value.
 * @return Returns a pointer to the scaled float representation of the data on the device. Note that this function
 * allocated memory on the device which has to be released later with `cudaFree` when the memory is no longer required.
 * Before deleting the memory it has to be ensured that all operations on it using a asynchronous CUDA stream have to be
 * completed.
 */
__host__ float *scaleWavData(const int *dDataInt,
                             const cudaStream_t &stream,
                             const size_t dataSize,
                             const float scaleVal) {
    // Now copy into a float array and scale the values (use kernel for this)
    float *dDataFloat;
    checkCuda(cudaMalloc((void **) &dDataFloat, dataSize * sizeof(float))); // device
    auto [block, grid] = getBlockGridSize(dataSize);
    divIntByFloat<<<grid, block, 0, stream>>>(dDataInt, dDataFloat, scaleVal, dataSize);
    checkCuda(cudaGetLastError());
    return dDataFloat;
}

/*!
 * Down-sample a signal by a specified factor. Note that not anti-aliasing filtering or similar is performed in advance
 * to prevent aliasing or folding effects. It has to be ensured that high frequencies are removed sufficiently before
 * down-sampling.
 * @param dSignalToBeDecimated Input signal on the device
 * @param stream CUDA stream used for the operation
 * @param decimationFactor Factor by which the signal shall be down-sampled.
 * @param dataSize Total length of the signal/sequence
 * @return Returns a down-sampled signal. Note that the down-sampled signal is stored in a new array on the device that
 * was allocated for this purpose. Hence, the memory has to be freed using `cudaFree`, once it is no longer required.
 */
__host__ std::tuple<float *, size_t> decimateSignal(const float *const dSignalToBeDecimated,
                                                    const cudaStream_t &stream,
                                                    const int decimationFactor,
                                                    const size_t dataSize) {
    size_t decimatedSize = dataSize / decimationFactor;
    float *dSignalDecimated;
    checkCuda(cudaMalloc((void **) &dSignalDecimated, decimatedSize * sizeof(float)));
    auto [block, grid] = getBlockGridSize(dataSize);
    decimate<<<grid, block, 0, stream>>>(dSignalToBeDecimated, dSignalDecimated, decimationFactor, dataSize);
    checkCuda(cudaGetLastError());

    return {dSignalDecimated, decimatedSize};
}

/*!
 * Compute the power of a signal. The power of a signal is the sum of the absolute squares of its time-domain
 * samples divided by the signal length. We will use this signal power later as reference for the filtered
 * signals in order to decide whether a particular frequency is present or not!
 * @param dSignal Input signal on the device
 * @param stream CUDA stream used for the operation
 * @param dataSize Total length of the signal
 * @return Pointer to the signal power value. Note that the corresponding memory has to be released again later using
 * cudaFree()` when no longer needed.
 */
__host__ float *getSignalPower(const float *const dSignal, const cudaStream_t &stream, const size_t dataSize) {
    float *dSignalPow2;
    auto [block, grid] = getBlockGridSize(dataSize);
    checkCuda(cudaMalloc((void **) &dSignalPow2, dataSize * sizeof(float)));

    // First square the signal
    squared<<<grid, block, 0, stream>>>(dSignal, dSignalPow2, dataSize);
    checkCuda(cudaGetLastError());

    // In each iteration, we can reduce the number of threads by a factor of 2. Hence, we can collect the block & grid
    // dimensions here in this array and then later iterate through the array, dividing the elements by 2
    unsigned int *dims[] = {&grid.x, &grid.y, &grid.z, &block.x, &block.y, &block.z};
    for (int factor = 1, j = 0; factor < dataSize; factor <<= 1) {
        *dims[j++ % 6] >>= 1;
        sum<<<grid, block, 0, stream>>>(dSignalPow2, factor, dataSize);
        checkCuda(cudaGetLastError());
    } // The sum will be in the leftmost element of the array

    // Compute mean:
    float *dSqSignalMean = &dSignalPow2[0];
    divFloat<<<1, 1, 0, stream>>>(dSqSignalMean, dataSize, 1);
    checkCuda(cudaGetLastError());

    return dSqSignalMean;
}

/*!
 * Filter a given signal using bandpass filters for various bandpass frequencies. All filtered signals will be returned
 * in an array of size `dataSize` * freqz.size()`.
 * @param dSignal Input signal that will be filtered several times (depending on the size of `freqz`)
 * @param dataSize The length of the input signal
 * @param freqz All bandpass frequencies for which a bandpass filter is created. The input signal is filtered using each
 * bandpass filter.
 * @param L The filter length that shall be used for the bandpass filters
 * @param fs The sample frequency of the input signal
 * @param streams The input signal can be processed (filtered) in parallel using several streams. For each filter
 * frequency an individual CUDA stream is required to ensure that all filters can be applied simultaneously.
 * @return Returns
 *    1. An device array containing all filtered signals. All resulting signals are stacked horizontally in
 *    the one-dimensional output array in the order of the frequencies in `freqz`. The overall size of the output array
 *    is `dataSize` * freqz.size()`.  |bandpass_filtered_signal_1|bandpass_filtered_signal_2|...
 *    2. A vector of pointers to pinned host memory, containing the FIR filter weights
 *    3. A vector of pointers to device memory, containing the FIR filter weights
 *    Note that the memory areas for the pointers in 2 & 3 have to be freed using `cudaHostFree()` and `cudaFree()`
 *    when they are not required any more.
 */
__host__ std::tuple<float *, std::vector<float *>, std::vector<float *>>
filterSignal(const float *const dSignal,
             const size_t dataSize,
             const std::vector<float> &freqz,
             const int L,
             const float fs,
             const std::vector<cudaStream_t> &streams) {
    std::vector<float *> hhh(freqz.size());
    std::vector<float *> dhh(freqz.size());
    float *dOutConvolve;
    checkCuda(cudaMalloc(&dOutConvolve, dataSize * freqz.size() * sizeof(float)));

    auto [block, grid] = getBlockGridSize(dataSize);

    // Filter the signal using different CUDA streams
    for (int i = 0; i < freqz.size(); i++) {
        auto fb = freqz[i];
        //checkCuda(cudaStreamCreate(&streams[i]));//
        checkCuda(cudaMalloc((void **) &dhh[i], L * sizeof(float)));
        checkCuda(cudaMallocHost((void **) &hhh[i], L * sizeof(float)));
        auto hh = filterCoeff(fb, fs, L);
        std::copy(hh.begin(), hh.end(), hhh[i]); // copy to pinned memory
        checkCuda(cudaMemcpyAsync(dhh[i], hhh[i], L * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        convolve<<<grid, block, 0, streams[i]>>>(dSignal, dOutConvolve + i * dataSize, dhh[i], L, dataSize);
        checkCuda(cudaGetLastError());
    }
    return {dOutConvolve, hhh, dhh};
}

/*!
 * Score a signal which was squared and where the signal power was computed in a sliding window. The windowed
 * signal power values are compared with a reference value. If the values exceed the reference a binary flag is
 * set in the output signal.
 * @param dWindowedPowerSignal An array containing `streams.size()` subsequences. The subsequences are organized in the
 * array as |subsequence_1|subsequence_2|...|subsequence_n| where `n = streams.size()`. The scoring is performed in
 * parallel for all subsequences using different streams
 * @param signalSize The length of a subsequence
 * @param dSigPowerRef The reference signal power value that is used as threshold
 * @param streams A vector of CUDA streams. For each subsequence an individual stream is used to perform the operation.
 * @return An array containing `streams.size()` subsequences. The subsequences are organized in the
 * array as |subsequence_1|subsequence_2|...|subsequence_n| where `n = streams.size()`. Each output subsequence contains
 * binary values (0,1) indicating whether or not the reference threshold was exceeded at a particular instance of time
 * or not. Note that this returned vector points to an allocated device memory area which has to be released using
 * `cudaFree`, when it is no longer needed.
 */
__host__ int *scoreSignalPower(const float *const dWindowedPowerSignal,
                               const size_t signalSize,
                               const float *dSigPowerRef,
                               const std::vector<cudaStream_t> &streams) {
    int *dActiveFreq;
    checkCuda(cudaMalloc(&dActiveFreq, signalSize * streams.size() * sizeof(int)));
    //checkCuda(cudaMemset(dActiveFreq, 0, signalSize * streams.size() * sizeof(int))); // Is this necessary?

    // Check, if the squared signals exceed a given value (power of the original dial tone signal)
    auto [block, grid] = getBlockGridSize(signalSize);
    for (int i = 0; i < streams.size(); i++) {
        auto din = dWindowedPowerSignal + i * signalSize;
        auto dOut = dActiveFreq + i * signalSize;

        // Then check, if it exceeds the average original signal power
        largerThan<<<grid, block, 0, streams[i]>>>(din, dOut, dSigPowerRef, signalSize);
        checkCuda(cudaGetLastError());
    }
    return dActiveFreq;
}

/*!
 * Filter an input signal using several bandpass filters and subsequently highlight the active frequencies in the
 * filtered signals by setting the corresponding values in the output sequences to the particular
 * frequency value.
 * @param dSignal The input signal containing the WAV audio data with the DTMF ringtones
 * @param dataSize The overall length of the input signal
 * @param freqz A list of frequencies, for which we want to create a bandpass filter and then filter the input signal
 * @param L The length that we want to use for the bandpass filter
 * @param fs The sample frequency of the input signal
 * @param dSigPowerRef The signal power of the input signal (possibly, scaled by a certain factor) that is used as
 *        reference. If the values in a filtered signal exceed this threshold, we assume that the corresponding
 *        bandpass frequency is active and the output is marked accordingly.
 * @return
 *        An output  array of size [`dataSize` x `n`] where `n=streams.size()`. The input signal is filtered `n` times
 *        using the `n` different bandpass filters. Each bandpass filter will remove all frequencies except one from
 *        the signal. Subsequently, the remaining "active" areas in the filtered signals are highlighted by setting
 *        them to the corresponding frequency value. The output array is organized as
 *        |subsequence_1|subsequence_2|...|subsequence_n| where in `subsequence_1` the areas in which the 1st frequency is
 *        active are highlighted, in `subsequence_2` the areas with containing the 2nd frequency are highlighted, and
 *        so on.
 *        Note that the memory for this output array was allocated inside this function. So, when the result is no
 *        longer needed, it has to be deleted using the `cudaFree()` function.
 */
__host__ int *filterSignalAndScoreFreqs(
        const float *const dSignal,
        const size_t dataSize,
        const std::vector<float> &freqz,
        const int L,
        const float fs,
        const float *const dSigPowerRef) {
    std::vector<cudaStream_t> streams(freqz.size());
    for (auto &stream: streams) {
        checkCuda(cudaStreamCreate(&stream));
    }

    // Filter the signal using bandpass filters for the specified frequencies
    std::vector<void *> pinnedMemCollection;
    std::vector<void *> deviceMemCollection;
    auto [dOutConvolve, hhh, dhh] = filterSignal(dSignal, dataSize, freqz, L, fs, streams);
    pinnedMemCollection.insert(pinnedMemCollection.end(), hhh.begin(), hhh.end());
    deviceMemCollection.insert(deviceMemCollection.end(), dhh.begin(), dhh.end());
    deviceMemCollection.emplace_back(dOutConvolve);

    // square all the signals in dOutConvolve
    squareSignals(dOutConvolve, streams, dataSize);

    // Then compute the signal power of the squared filtered signal in a sliding window
    auto [dhhAvg, dOutPower] = slidingWinSignalPower(dOutConvolve, streams, dataSize);
    deviceMemCollection.push_back(dhhAvg);
    deviceMemCollection.push_back(dOutPower);

    // Threshold all filtered (and squared) signals against the signal power of the original signal. All frequencies
    // outside the pass bands should be removed.
    int *dActiveFreq = scoreSignalPower(dOutPower, dataSize, dSigPowerRef, streams);
    //deviceMemCollection.emplace_back(dActiveFreq);

    // now multiply the binary signals with the corresponding frequencies
    auto [block, grid] = getBlockGridSize(dataSize);
    for (int i = 0; i < freqz.size(); i++) {//
        auto dInout = dActiveFreq + i * dataSize;
        multInt<<<grid, block, 0, streams[i]>>>(dInout, freqz[i], dataSize);
        checkCuda(cudaGetLastError());
    }
    for (int i = 0; i < freqz.size(); i++) {
        checkCuda(cudaStreamSynchronize(streams[i])); // synchronize all streams, since we need all results now...
        checkCuda(cudaStreamDestroy(streams[i])); // and destroy all streams. We do not need them any longer…
    }

    // We can free the pinned host memory already
    for (const auto p: pinnedMemCollection) {
        cudaFreeHost(p);
    }

    // We can also free the device memory already, since we synchronized all streams
    for (const auto p: deviceMemCollection) {
        cudaFree(p);
    }

    return dActiveFreq;
}

/*!
 * Finds the largest 2 frequencies among a set of `numDimsIn` subsequences of equal length along the time axis. Each
 * subsequence corresponds to a particular frequency and the areas where the frequency is active are set to its value
 * in that subsequence.
 * Hence, an input sequence with the shape [`signalSize` x `numDimsIn`] is reduced to a sequence of
 * shape [`numElements` x 2]. If there are 2 frequencies active at a time, they will both appear in the output
 * sequence (the lower frequency will be in the first subsequence and the higher frequency in the second subsequence).
 * This function is used to identify the two frequencies, which characterize a symbol in the DTMF signal.
 * @param dInputSignals An array containing `n=numSubSignals` subsequences. The subsequences are organized in the
 * array as |subsequence_1|subsequence_2|...|subsequence_n|. For each instance of time the function looks at the values
 * in the `n` subsequences and selects the 2 largest ones, which it writes to the 2-dimensional output sequence.
 * @param signalSize The length of an individual subsequence.
 * @param numSubSignals The number of sub-signals/subsequences in `dInputSignals`
 * @param stream The CUDA stream on which we want to operate
 * @return The function runs through all passed subsequences in the input and in each time step selects the two largest
 *   values among all subsequences which it writes to the 2-dimensional output sequence. The returned output sequence
 *   has, therefore, the shape [`numElements` x 2]. Since the memory for this array is allocated in this function, it
 *   will have to be freed later using `cudaFree()` when it is not longer needed.
 */
__host__ int *getTop2Frequencies(const int *const dInputSignals,
                                 const size_t signalSize,
                                 const size_t numSubSignals,
                                 const cudaStream_t &stream) {
    int *dTop2;
    checkCuda(cudaMalloc(&dTop2, signalSize * 2 * sizeof(int)));
    //checkCuda(cudaMemset(dTop2, 0, signalSize * 2 * sizeof(int))); // probably not really necessary...

    auto [block, grid] = getBlockGridSize(signalSize);
    top2<<<grid, block, 0, stream>>>(dInputSignals, dTop2, numSubSignals, signalSize);
    checkCuda(cudaGetLastError());
    return dTop2;
}

/*!
 * Perform max pooling on an input array. The function looks at adjacent pairs of values and selects the maximum in
 * each pair and writes it to the output array.
 * At a later stage in the data processing, we have longer sequences containing the frequency values, which were
 * originally present in the signal. Since we do not care about the length of these sequences (a subsequence containing
 * the frequencies 697 Hz and 1209 Hz will represent the symbol "1", no matter if it has a length of 100 or 1000). Hence,
 * we perform several pooling steps (in each step the sequence length is halved), to significantly reduce the final
 * computation tasks.
 * @param dSignal Input signal which shall be pooled
 * @param signalSize The size of the input signal
 * @param stream The CUDA stream to operate on
 * @return A 3-tuple containing the following return values:
 *         1. The pooled signal which should be significantly shorter than the input signal
 *         2. The length of the above pooled signal
 *         3. A vector containing device memory pointers, which have to be deleted later using `cudaFree()`, since
 *            we cannot ensure that the tasks on the stream have yet completed on completion of the function.
 */
__host__ std::tuple<const int *, size_t, std::vector<void *>> maxPoolSignal(const int *dSignal,
                                                                            const size_t signalSize,
                                                                            const cudaStream_t &stream) {
    std::vector<void *> deviceMemCollection;
    size_t pooledSize = signalSize << 1;// mult. by 2 since it will divide by 2 at the beginning of the for-loop again.
    for (int i = 0; i < NUM_POOLING_STEPS; i++) {
        pooledSize >>= 1;
        auto [block, grid] = getBlockGridSize(pooledSize);
        int *dTop2Pooled;
        checkCuda(cudaMalloc(&dTop2Pooled, pooledSize * sizeof(int)));
        deviceMemCollection.push_back(dTop2Pooled);
        maxPool<<<grid, block, 0, stream>>>(dSignal, dTop2Pooled, pooledSize);
        checkCuda(cudaGetLastError());
        dSignal = dTop2Pooled;
    }
    return {dSignal, pooledSize, deviceMemCollection};
}

/*!
 * From a 2-dimensional input sequence we can now infer the dialed numbers.
 * @param hTop2 The input sequence with shape [`pooledSize` x 2]. Since each dialed symbol is represented by 2
 * frequencies, we run through this 2-dimensional input sequence and map the 2-dimensional frequency vectors that we
 * encounter on the way to a symbol. Symbols are separated by a break ( [0,0]-vector in the input sequence). Hence, if a
 * frequency vector repeats before we recognize a [0,0]-vector, we ignore it. Only after observing a pause, we are ready
 * again to detect the next symbol in the sequence.
 * @param pooledSize The size of a subsequence in `hTop2`
 * @return The dialed number/symbol sequence as a string.
 */
__host__ std::string getDialedSequence(const int *const hTop2, const size_t pooledSize) {
    // Now do something similar to run-length encoding and collect all frequency pairs
    std::vector<std::pair<int, int>> allFreqPairs;
    bool waitForPause = false;
    for (int i = 0; i < pooledSize / 2; i++) {
        int row = hTop2[i];
        int col = hTop2[i + pooledSize / 2];
        if (row > 0 && col > 0 && !waitForPause) {
            std::pair<int, int> freqPair{row, col};
            allFreqPairs.push_back(freqPair);
            waitForPause = true;
        } else if (hTop2[i] == 0) { // We reached a pause, so now a new symbol can be accepted again...
            waitForPause = false;
        }
    }

    std::string sequence;
    for (const auto &pair: allFreqPairs) {
        sequence += freqToSymbolMap.at(pair.first).at(pair.second);
    }
    return sequence;
}

/*!
 * Clean up the stream used, the remaining pinned host memory and the device memory.
 * @param stream The CUDA stream to be destroyed
 * @param pinnedMemCollection A list of pointers to pinned host memory which shall be deleted
 * @param deviceMemCollection A list of pointers to the device memory which shall be deleted
 */
__host__ void cleanUp(cudaStream_t &stream, std::vector<void *> &pinnedMemCollection,
                      std::vector<void *> &deviceMemCollection) {
    // Destroy the CUDA stream
    checkCuda(cudaStreamDestroy(stream));

    // Free device memory
    for (auto p: deviceMemCollection) {
        checkCuda(cudaFree(p));
    }

    // Free pinned host memory
    for (auto p: pinnedMemCollection) {
        checkCuda(cudaFreeHost(p));
    }

    // Reset the device
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    checkCuda(cudaDeviceReset());
}

std::string processFile(const std::string &filePath) {
    // Read the WAV file and get the audio data + the WAV header
    auto [wavHeader, data] = readWAVFile(filePath);

    // Create a stream for copying the data to the device and other operations on the device
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));

    // Collect device memory and host pinned memory in these vectors
    std::vector<void *> pinnedMemCollection;
    std::vector<void *> deviceMemCollection;

    // Copy the audio wav file to the device memory
    auto [hPinnedDataInt, dDataInt] = copyWavDataToDevice(data, stream);
    pinnedMemCollection.push_back(hPinnedDataInt);
    deviceMemCollection.push_back(dDataInt);

    // Since the Data in the WAV file is integer format, let us scale it and put it into a float format
    // It is not really necessary to scale the data, but it might look a bit nicer...
    // 2147483647 = 2^31 - 1 is the largest value, representable by an 4-byte signed integer
    auto dDataFloat = scaleWavData(dDataInt, stream, data.size(), 2147483647.0F);
    deviceMemCollection.push_back(dDataFloat);

    // Now decimate the signal by a given factor (here: 8). This could also be done before scaling all elements
    // of the signal
    auto [dSignal, decimatedSize] = decimateSignal(dDataFloat, stream, DECIMATION_FACTOR, data.size());
    deviceMemCollection.push_back(dSignal);


    //Get Signal Power: The power of a signal is the sum of the absolute squares of its time-domain
    // samples divided by the signal length. We will use this signal power later than reference for the filtered
    // signals in order to decide whether a particular frequency is present or not!
    auto dSqSignalMean = getSignalPower(dSignal, stream, decimatedSize);
    deviceMemCollection.push_back(dSqSignalMean);

    // Divide the signal power by a factor of 2. This works better in practice when being used as a threshold later
    // In this example, values between 2.0 & 20.0 appear to work well
    divFloat<<<1, 1, 0, stream>>>(dSqSignalMean, 2.0F, 1);
    checkCuda(cudaGetLastError());

    // Wait for work on previous stream to complete, since we will create a bunch of new streams in the following!
    // Will likely be completed already now...
    checkCuda(cudaStreamSynchronize(stream));


    // Filter the signal with the individual bandpass filters. Subsequently, highlight the individual frequencies
    // in the signal areas.
    auto dActiveFreq = filterSignalAndScoreFreqs(
            dSignal,
            decimatedSize,
            DTMF_FREQUENCIES,
            FILTER_LENGTH,
            float(wavHeader.SamplesPerSec) / float(DECIMATION_FACTOR),
            dSqSignalMean);
    deviceMemCollection.push_back(dActiveFreq);

    // We have in total 8 signals as indicator for the 8 frequencies that might be active. However, only 2
    // frequencies can be active at the same instance of time. Hence,we look for the 2 strongest frequencies
    // and end up with 2 signals instead of 8, representing the 2 active frequencies (if there are any)
    // at any instance of time
    int *dTop2 = getTop2Frequencies(dActiveFreq, decimatedSize, DTMF_FREQUENCIES.size(), stream);
    deviceMemCollection.push_back(dTop2);


    // Since not all values are set where two frequencies are active (because we did some simple thresholding in time
    // domain using the input signal power as threshold) we have to now do some sort of max pooling. This means that
    // we look at temporally adjacent frequency values and select the larger one (the one unequal zero). At the same
    // time we incrementally reduce the length of the 2-dimensional sequence, since it is (almost) irrelevant how long a number
    // is "pressed". What counts is that the pressed numbers are separated by a break. By reducing the sequence length
    // we simplify the following processing step where we want to determine the overall sequence of symbols.
    auto [dTop2Pooled, pooledSize, dMems] = maxPoolSignal(dTop2, decimatedSize, stream);
    deviceMemCollection.insert(deviceMemCollection.end(), dMems.begin(), dMems.end());

    // Copy the pooled array back to the host
    int *hTop2;
    checkCuda(cudaMallocHost((void **) &hTop2, pooledSize * sizeof(int))); // host pinned
    pinnedMemCollection.emplace_back(hTop2);
    checkCuda(cudaMemcpyAsync(hTop2, dTop2Pooled, pooledSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Print the result to stdout
    std::string sequence = getDialedSequence(hTop2, pooledSize);
    if(VERBOSE > 1) {
        std::cout << "Your dialed sequence: ";
        for (const auto &c: sequence) {
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }
    //std::cout << "Expected sequence:    ";
    //std::cout << "A A * 5 4 8 8 A 4 6 * 4 5 3 1 7 8 8 2 7 A 2 # 3 1 C 2 4 0 3 5 9 4 6 1 8 0 7 C C A 2 C 9 0 D 5 # 4 5" << std::endl;
    cleanUp(stream, pinnedMemCollection, deviceMemCollection);

    return sequence;
}

void replaceAll( std::string &s, const std::string &search, const std::string &replace ) {
    for( size_t pos = 0; ; pos += replace.length() ) {
        // Locate the substring to replace
        pos = s.find( search, pos );
        if( pos == std::string::npos ) break;
        // Replace by erasing and inserting
        s.erase( pos, search.length() );
        s.insert( pos, replace );
    }
}

__host__ int main(int argc, char *argv[]) {
    std::string filePath;
    if (argc <= 1) {
        filePath = "../wav/dial3.wav";
    } else {
        filePath = argv[1];
        std::cout << "Input wave file name: " << filePath << std::endl;
    }

    if (fs::is_regular_file(filePath)) {
        // Only one file
        processFile(filePath);
    } else if (fs::is_directory(filePath)) {
        // A directory, loop through all *.wav files
        int i = 1;
        for (auto const &dir_entry: fs::directory_iterator{fs::path{filePath}}) {
            if(dir_entry.path().extension() != ".wav") continue;
            std::string fileName = dir_entry.path().filename().string();

            // In this example we only want those sequences, where the file name matches the actual sequence
            // Comment/remove this line, if you also want to process other files in the directory...
            if(fileName.length() < 24) continue;

            std::cout << std::setfill(' ') << std::setw(4) << i << "." << " File name (.wav):  " << fileName << ", ";

            replaceAll(fileName, ".wav", "");
            replaceAll(fileName, "S", "*");
            replaceAll(fileName, "H", "#");
            std::cout << "\n" << "      File name cleaned: " << fileName << "\n";

            std::string seq = processFile(dir_entry.path());
            std::cout << "      Algorithm result:  " << seq << "\n";
            std::cout << "      Are file name and algorithm result the same?: " <<
                         (seq == fileName ? "true" : "false") << "\n";
            if(seq != fileName) {
                std::cerr << "Error: File name and algorithm result did not match: Aborting!";
                exit(EXIT_FAILURE);
            }
            std::cout << std::endl;

            ++i;
        }
    }


    return 0;
}
