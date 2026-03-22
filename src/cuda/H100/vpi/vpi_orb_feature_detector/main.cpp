/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/ImageFlip.h>
#include <vpi/algo/ORB.h>

#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

static cv::Mat DrawKeypoints(cv::Mat img, VPIPyramidalKeypointF32 *kpts, VPIBriefDescriptor *descs, int numKeypoints)
{
    cv::Mat out;
    img.convertTo(out, CV_8UC1);
    cvtColor(out, out, cv::COLOR_GRAY2BGR);

    if (numKeypoints == 0)
    {
        return out;
    }

    std::vector<int> distances(numKeypoints, 0);
    float maxDist = 0.f;

    for (int i = 0; i < numKeypoints; i++)
    {
        for (int j = 0; j < VPI_BRIEF_DESCRIPTOR_ARRAY_LENGTH; j++)
        {
            distances[i] += std::bitset<8 * sizeof(uint8_t)>(descs[i].data[j] ^ descs[0].data[j]).count();
        }
        if (distances[i] > maxDist)
        {
            maxDist = distances[i];
        }
    }

    uint8_t ids[256];
    std::iota(&ids[0], &ids[0] + 256, 0);
    cv::Mat idsMat(256, 1, CV_8UC1, ids);

    cv::Mat cmap;
    applyColorMap(idsMat, cmap, cv::COLORMAP_JET);

    for (int i = 0; i < numKeypoints; i++)
    {
        int cmapIdx = static_cast<int>(std::round((distances[i] / maxDist) * 255));

        float rescale = std::pow(2, kpts[i].octave);
        float x       = kpts[i].x * rescale;
        float y       = kpts[i].y * rescale;

        circle(out, cv::Point(x, y), 3, cmap.at<cv::Vec3b>(cmapIdx, 0), -1);
    }

    return out;
}

int main(int argc, char *argv[])
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    cv::Mat cvImage;

    // VPI objects that will be used
    VPIImage imgInput     = NULL;
    VPIImage imgGrayScale = NULL;

    VPIPyramid pyrInput   = NULL;
    VPIArray keypoints    = NULL;
    VPIArray descriptors  = NULL;
    VPIPayload orbPayload = NULL;
    VPIStream stream      = NULL;

    int retval = 0;

    try
    {
        // =============================
        // Parse command line parameters

        if (argc != 3)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|cuda|pva> <input image>");
        }

        std::string strBackend       = argv[1];
        std::string strInputFileName = argv[2];

        // Now parse the backend
        VPIBackend backend;

        if (strBackend == "cpu")
        {
            backend = VPI_BACKEND_CPU;
        }
        else if (strBackend == "cuda")
        {
            backend = VPI_BACKEND_CUDA;
        }
        else if (strBackend == "pva")
        {
            backend = VPI_BACKEND_PVA;
        }
        else
        {
            throw std::runtime_error("Backend '" + strBackend +
                                     "' not recognized, it must be either cpu, cuda or pva.");
        }

        // Use the selected backend with CPU to be able to read data back from CUDA to CPU for example.
        const VPIBackend backendWithCPU = static_cast<VPIBackend>(backend | VPI_BACKEND_CPU);

        // =====================
        // Load the input image

        cvImage = cv::imread(strInputFileName);
        if (cvImage.empty())
        {
            throw std::runtime_error("Can't open first image: '" + strInputFileName + "'");
        }

        // =================================
        // Allocate all VPI resources needed

        // Create the stream where processing will happen
        CHECK_STATUS(vpiStreamCreate(0, &stream));

        // Define the algorithm parameters.
        VPIORBParams orbParams;
        CHECK_STATUS(vpiInitORBParams(&orbParams));

        orbParams.fastParams.intensityThreshold = 142;
        orbParams.maxFeaturesPerLevel           = 88;
        orbParams.maxPyramidLevels              = 3;

        // PVA backend doesn't support default Harris Corner Scoring
        if (backend == VPI_BACKEND_PVA)
        {
            orbParams.scoreType = VPI_CORNER_SCORE_FAST;
        }

        // We now wrap the loaded image into a VPIImage object to be used by VPI.
        // VPI won't make a copy of it, so the original image must be in scope at all times.
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImage, 0, &imgInput));
        CHECK_STATUS(vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &imgGrayScale));

        // For the output arrays capacity we can use the maximum number of features per level multiplied by the
        // maximum number of pyramid levels, this will be the de factor maximum for all levels of the input.
        int outCapacity = orbParams.maxFeaturesPerLevel * orbParams.maxPyramidLevels;

        // Create the output keypoint array.
        CHECK_STATUS(vpiArrayCreate(outCapacity, VPI_ARRAY_TYPE_PYRAMIDAL_KEYPOINT_F32, backendWithCPU, &keypoints));

        // Create the output descriptors array.  To output corners only use NULL instead.
        CHECK_STATUS(vpiArrayCreate(outCapacity, VPI_ARRAY_TYPE_BRIEF_DESCRIPTOR, backendWithCPU, &descriptors));

        // For the internal buffers capacity we can use the maximum number of features per level multiplied by 20.
        // This will make FAST find a large number of corners so then ORB can select the top N corners in
        // accordance to Harris score of each corner, where N = maximum number of features per level.
        int bufCapacity = orbParams.maxFeaturesPerLevel * 20;

        // Create the payload for ORB Feature Detector algorithm
        CHECK_STATUS(vpiCreateORBFeatureDetector(backend, bufCapacity, &orbPayload));

        // ================
        // Processing stage

        // First convert input to grayscale
        VPIBackend convertBackend = (backend == VPI_BACKEND_PVA) ? VPI_BACKEND_CPU : backend;
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, convertBackend, imgInput, imgGrayScale, NULL));

        // Then, create the Gaussian Pyramid for the image and wait for the execution to finish
        CHECK_STATUS(vpiPyramidCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, orbParams.maxPyramidLevels, 0.5,
                                      backend, &pyrInput));
        CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(stream, backend, imgGrayScale, pyrInput, VPI_BORDER_CLAMP));

        // Then get ORB features and wait for the execution to finish
        CHECK_STATUS(vpiSubmitORBFeatureDetector(stream, backend, orbPayload, pyrInput, keypoints, descriptors,
                                                 &orbParams, VPI_BORDER_CLAMP));

        CHECK_STATUS(vpiStreamSync(stream));

        // =======================================
        // Output processing and saving it to disk

        // Lock output keypoints and scores to retrieve its data on cpu memory
        VPIArrayData outKeypointsData;
        VPIArrayData outDescriptorsData;
        VPIImageData imgData;
        CHECK_STATUS(vpiArrayLockData(keypoints, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outKeypointsData));
        CHECK_STATUS(vpiArrayLockData(descriptors, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outDescriptorsData));
        CHECK_STATUS(vpiImageLockData(imgGrayScale, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData));

        VPIPyramidalKeypointF32 *outKeypoints = (VPIPyramidalKeypointF32 *)outKeypointsData.buffer.aos.data;
        VPIBriefDescriptor *outDescriptors    = (VPIBriefDescriptor *)outDescriptorsData.buffer.aos.data;

        cv::Mat img;
        CHECK_STATUS(vpiImageDataExportOpenCVMat(imgData, &img));

        // Draw the keypoints in the output image
        cv::Mat outImage = DrawKeypoints(img, outKeypoints, outDescriptors, *outKeypointsData.buffer.aos.sizePointer);

        // Save the output image to disk
        imwrite("orb_feature_detector_" + strBackend + ".png", outImage);

        // Done handling outputs, don't forget to unlock them.
        CHECK_STATUS(vpiImageUnlock(imgGrayScale));
        CHECK_STATUS(vpiArrayUnlock(keypoints));
        CHECK_STATUS(vpiArrayUnlock(descriptors));
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // ========
    // Clean up

    // Make sure stream is synchronized before destroying the objects
    // that might still be in use.
    vpiStreamSync(stream);

    vpiImageDestroy(imgInput);
    vpiImageDestroy(imgGrayScale);
    vpiArrayDestroy(keypoints);
    vpiArrayDestroy(descriptors);
    vpiPayloadDestroy(orbPayload);
    vpiStreamDestroy(stream);

    return retval;
}
