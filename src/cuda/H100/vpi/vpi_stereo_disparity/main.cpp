/*
* Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION >= 3
#    include <opencv2/imgcodecs.hpp>
#else
#    include <opencv2/contrib/contrib.hpp> // for colormap
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include <cstring>
#include <iostream>
#include <sstream>

#define CHECK_STATUS(STMT)                                                                  \
    do                                                                                      \
    {                                                                                       \
        VPIStatus status = (STMT);                                                          \
        if (status != VPI_SUCCESS)                                                          \
        {                                                                                   \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];                                     \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));                                \
            std::ostringstream ss;                                                          \
            ss << "line " << __LINE__ << " " << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());                                             \
        }                                                                                   \
    } while (0);

int main(int argc, char *argv[])
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    cv::Mat cvImageLeft, cvImageRight;

    // VPI objects that will be used
    VPIImage inLeft        = NULL;
    VPIImage inRight       = NULL;
    VPIImage tmpLeft       = NULL;
    VPIImage tmpRight      = NULL;
    VPIImage stereoLeft    = NULL;
    VPIImage stereoRight   = NULL;
    VPIImage disparity     = NULL;
    VPIImage confidenceMap = NULL;
    VPIStream stream       = NULL;
    VPIPayload stereo      = NULL;

    int retval = 0;

    try
    {
        // =============================
        // Parse command line parameters

        if (argc != 4)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] +
                                     " <cuda|ofa|ofa-pva-vic> <left image> <right image>");
        }

        std::string strBackend       = argv[1];
        std::string strLeftFileName  = argv[2];
        std::string strRightFileName = argv[3];

        uint64_t backends;

        if (strBackend == "cuda")
        {
            backends = VPI_BACKEND_CUDA;
        }
        else if (strBackend == "ofa")
        {
            backends = VPI_BACKEND_OFA;
        }
        else if (strBackend == "ofa-pva-vic")
        {
            backends = VPI_BACKEND_OFA | VPI_BACKEND_PVA | VPI_BACKEND_VIC;
        }
        else
        {
            throw std::runtime_error("Backend '" + strBackend +
                                     "' not recognized, it must be either cuda, ofa or ofa-pva-vic.");
        }

        // =====================
        // Load the input images
        cvImageLeft = cv::imread(strLeftFileName);
        if (cvImageLeft.empty())
        {
            throw std::runtime_error("Can't open '" + strLeftFileName + "'");
        }

        cvImageRight = cv::imread(strRightFileName);
        if (cvImageRight.empty())
        {
            throw std::runtime_error("Can't open '" + strRightFileName + "'");
        }

        // =================================
        // Allocate all VPI resources needed

        int32_t inputWidth  = cvImageLeft.cols;
        int32_t inputHeight = cvImageLeft.rows;

        // Create the stream that will be used for processing.
        CHECK_STATUS(vpiStreamCreate(0, &stream));

        // We now wrap the loaded images into a VPIImage object to be used by VPI.
        // VPI won't make a copy of it, so the original image must be in scope at all times.
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImageLeft, 0, &inLeft));
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImageRight, 0, &inRight));

        // Format conversion parameters needed for input pre-processing
        VPIConvertImageFormatParams convParams;
        CHECK_STATUS(vpiInitConvertImageFormatParams(&convParams));

        // Initialize default parameters
        VPIStereoDisparityEstimatorCreationParams createParams;
        CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&createParams));

        // Select max disparity that works well for the chair_stereo_{left,right}_1920.png files
        createParams.maxDisparity = 256;

        // Default format and size for input stereo pair (some backends require adjustments, see below)
        VPIImageFormat stereoFormat = VPI_IMAGE_FORMAT_Y8_ER;

        int stereoWidth  = inputWidth;
        int stereoHeight = inputHeight;

        // Default format and size for output
        VPIImageFormat disparityFormat = VPI_IMAGE_FORMAT_S16;

        int outputWidth  = inputWidth;
        int outputHeight = inputHeight;

        // Override some backend-dependent parameters
        if (strBackend.find("ofa") != std::string::npos)
        {
            // Implementations using OFA require BL input
            stereoFormat = VPI_IMAGE_FORMAT_Y8_ER_BL;

            if (strBackend == "ofa")
            {
                // when using OFA alone, output must also be BL
                disparityFormat = VPI_IMAGE_FORMAT_S16_BL;
            }

            // Using downscale factor with OFA improves performance
            createParams.downscaleFactor = 2;
            outputWidth  = (inputWidth + createParams.downscaleFactor - 1) / createParams.downscaleFactor;
            outputHeight = (inputHeight + createParams.downscaleFactor - 1) / createParams.downscaleFactor;

            // Output width including downscaleFactor must be at least max(64, maxDisparity/downscaleFactor) when the
            // OFA+PVA+VIC backend is used
            if (strBackend.find("pva") != std::string::npos)
            {
                int minWidth = std::max(createParams.maxDisparity / createParams.downscaleFactor, outputWidth);
                outputWidth  = std::max(64, minWidth);
                outputHeight = (inputHeight * outputWidth) / inputWidth;
                stereoWidth  = outputWidth * createParams.downscaleFactor;
                stereoHeight = outputHeight * createParams.downscaleFactor;
            }
        }

        // Create the payload for Stereo Disparity algorithm.
        // Payload is created before the image objects so that non-supported backends can be trapped with an error.
        CHECK_STATUS(vpiCreateStereoDisparityEstimator(backends, stereoWidth, stereoHeight, stereoFormat, &createParams,
                                                       &stereo));

        // Create the output image where the disparity map will be stored.
        CHECK_STATUS(vpiImageCreate(outputWidth, outputHeight, disparityFormat, 0, &disparity));

        // Create the input stereo images
        CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, stereoFormat, 0, &stereoLeft));
        CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, stereoFormat, 0, &stereoRight));

        // Create the confidence image if the backend can support it
        if (strBackend == "ofa-pva-vic" || strBackend == "cuda")
        {
            CHECK_STATUS(vpiImageCreate(outputWidth, outputHeight, VPI_IMAGE_FORMAT_U16, 0, &confidenceMap));
        }

        // If a rescale of the input is required, create temporary images for the initial format conversion.
        bool const isRescaleRequired = (stereoWidth != inputWidth) || (stereoHeight != inputHeight);
        if (isRescaleRequired)
        {
            CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, stereoFormat, 0, &tmpLeft));
            CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, stereoFormat, 0, &tmpRight));
        }

        // ================
        // Processing stage

        // Start with default parameters, and override some values depending on what backend is used.
        VPIStereoDisparityEstimatorParams submitParams;
        CHECK_STATUS(vpiInitStereoDisparityEstimatorParams(&submitParams));
        if (strBackend == "ofa-pva-vic")
        {
            // The INFERENCE confidence type achieves better performance with OFA+PVA+VIC backend. The only tradeoff is
            // that the deep-learning based confidence map is not easily expressed as a function of left and right
            // disparity estimates, in contrast to ABSOLUTE or RELATIVE confidence type.
            submitParams.confidenceType = VPI_STEREO_CONFIDENCE_INFERENCE;
        }
        else if (strBackend == "cuda")
        {
            // The chair_stereo_{left,right}_1920.png inputs benefit from a higher confidence threshold with CUDA
            submitParams.confidenceThreshold = UINT16_MAX - 10000;
        }

        // -----------------
        // Pre-process input
        if (isRescaleRequired)
        {
            // We require a conversion with CUDA only because we loaded the images in the default BGR format from OpenCV
            // and the VIC backend does not support 3-channel RGB/BGR image formats.
            // Alternatively, we could load grayscale images and handle the conversion+rescale in one operation on VIC.

            // Convert opencv input to grayscale format using CUDA
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inLeft, tmpLeft, &convParams));
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inRight, tmpRight, &convParams));

            // Rescale on VIC
            CHECK_STATUS(
                vpiSubmitRescale(stream, VPI_BACKEND_VIC, tmpLeft, stereoLeft, VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
            CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_VIC, tmpRight, stereoRight, VPI_INTERP_LINEAR,
                                          VPI_BORDER_CLAMP, 0));
        }
        else
        {
            // Convert opencv input to grayscale format using CUDA
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inLeft, stereoLeft, &convParams));
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inRight, stereoRight, &convParams));
        }

        // ------------------------------
        // Do stereo disparity estimation

        // Submit it with the input and output images
        CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream, backends, stereo, stereoLeft, stereoRight, disparity,
                                                       confidenceMap, &submitParams));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));

        // ========================================
        // Output pre-processing and saving to disk
        // Lock output to retrieve its data on cpu memory
        VPIImageData data;
        CHECK_STATUS(vpiImageLockData(disparity, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

        // Make an OpenCV matrix out of this image
        cv::Mat cvDisparity;
        CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cvDisparity));

        // Scale result and write it to disk. Disparities are in Q10.5 format,
        // so to map it to float, it gets divided by 32. Then the resulting disparity range,
        // from 0 to maxDisparity gets mapped to 0-255 for proper output.
        cvDisparity.convertTo(cvDisparity, CV_8UC1, 255.0 / (32 * createParams.maxDisparity), 0);

        // Apply JET colormap to turn the disparities into color.
        // Reddish hues represent objects closer to the camera, blueish are farther away.
        cv::Mat cvDisparityColor;
        applyColorMap(cvDisparity, cvDisparityColor, cv::COLORMAP_JET);

        // Done handling output, don't forget to unlock it.
        CHECK_STATUS(vpiImageUnlock(disparity));

        // If we have a confidence map, adjust it for display and write it to disk too.
        if (confidenceMap)
        {
            // Lock the image data and export to cv::Mat
            VPIImageData data;
            CHECK_STATUS(vpiImageLockData(confidenceMap, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));
            cv::Mat cvConfidence;
            CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cvConfidence));

            // Confidence map varies from 0 to 65535, we scale it to [0-255].
            cvConfidence.convertTo(cvConfidence, CV_8UC1, 255.0 / 65535, 0);
            imwrite("confidence_" + strBackend + ".png", cvConfidence);

            CHECK_STATUS(vpiImageUnlock(confidenceMap));

            // When pixel confidence is 0, we would like its color in the disparity image to be black.
            cv::Mat cvMask;
            threshold(cvConfidence, cvMask, 1, 255, cv::THRESH_BINARY);
            cvtColor(cvMask, cvMask, cv::COLOR_GRAY2BGR);
            bitwise_and(cvDisparityColor, cvMask, cvDisparityColor);
        }

        imwrite("disparity_" + strBackend + ".png", cvDisparityColor);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // ========
    // Clean up

    // Destroying stream first makes sure that all work submitted to
    // it is finished.
    vpiStreamDestroy(stream);

    // Only then we can destroy the other objects, as we're sure they
    // aren't being used anymore.

    vpiImageDestroy(inLeft);
    vpiImageDestroy(inRight);
    vpiImageDestroy(tmpLeft);
    vpiImageDestroy(tmpRight);
    vpiImageDestroy(stereoLeft);
    vpiImageDestroy(stereoRight);
    vpiImageDestroy(confidenceMap);
    vpiImageDestroy(disparity);
    vpiPayloadDestroy(stereo);

    return retval;
}
