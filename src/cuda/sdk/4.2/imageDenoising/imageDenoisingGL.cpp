/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



/*
 * This sample demonstrates two adaptive image denoising technqiues: 
 * KNN and NLM, based on computation of both geometric and color distance 
 * between texels. While both techniques are already implemented in the 
 * DirectX SDK using shaders, massively speeded up variation 
 * of the latter techique, taking advantage of shared memory, is implemented
 * in addition to DirectX counterparts.
 * See supplied whitepaper for more explanations.
 */


// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imageDenoising.h"

// includes, project
#include <rendercheck_gl.h> // automated testing
#include <sdkHelper.h>    // includes cuda.h and cuda_runtime_api.h
#include <shrQATest.h>    // standard utility and system includes

static char *sSDKsample = "CUDA ImageDenoising"; 

static char *filterMode[] = {
    "Passthrough",
    "KNN method",
    "NLM method",
    "Quick NLM(NLM2) method",
    NULL
};

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "image_passthru.ppm",
    "image_knn.ppm",
    "image_nlm.ppm",
    "image_nlm2.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_passthru.ppm",
    "ref_knn.ppm",
    "ref_nlm.ppm",
    "ref_nlm2.ppm",
    NULL
};

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
//Source image on the host side
uchar4 *h_Src;
int imageW, imageH;
GLuint shader;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int  g_Kernel = 0;
bool    g_FPS = false;
bool   g_Diag = false;
StopWatchInterface *timer = NULL;

//Algorithms global parameters
const float noiseStep = 0.025f;
const float  lerpStep = 0.025f;
static float knnNoise = 0.32f;
static float nlmNoise = 1.45f;
static float    lerpC = 0.2f;


const int frameN = 24;
int frameCounter = 0;


#define BUFFER_DATA(i) ((char *)0 + i)

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_AutoQuit = false;
bool g_Verify = false;
bool g_bQAReadback = false;
bool g_bOpenGLQA   = false;
bool g_bFBODisplay = false;

// CheckFBO/BackBuffer class objects
CFrameBufferObject  *g_FrameBufferObject = NULL;
CheckRender         *g_CheckRender       = NULL;

int *pArgc   = NULL;
char **pArgv = NULL;

#define MAX_EPSILON_ERROR 5
#define REFRESH_DELAY	  10 //ms

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

    // This will output the proper CUDA error strings in the event that a CUDA host call returns an error
    #define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

    inline void __checkCudaErrors( cudaError err, const char *file, const int line )
    {
        if( cudaSuccess != err) {
		    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                    file, line, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    // This will output the proper error string when calling cudaGetLastError
    #define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

    inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
    {
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                    file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    // General GPU Device CUDA Initialization
    int gpuDeviceInit(int devID)
    {
        int deviceCount;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
            exit(-1);
        }
        if (devID < 0) 
            devID = 0;
        if (devID > deviceCount-1) {
            fprintf(stderr, "\n");
            fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
            fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
            fprintf(stderr, "\n");
            return -devID;
        }

        cudaDeviceProp deviceProp;
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
        if (deviceProp.major < 1) {
            fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
            exit(-1);                                                  \
        }

        checkCudaErrors( cudaSetDevice(devID) );
        printf("> gpuDeviceInit() CUDA device [%d]: %s\n", devID, deviceProp.name);
        return devID;
    }

    // This function returns the best GPU (with maximum GFLOPS)
    int gpuGetMaxGflopsDeviceId()
    {
	    int current_device   = 0, sm_per_multiproc = 0;
	    int max_compute_perf = 0, max_perf_device  = 0;
	    int device_count     = 0, best_SM_arch     = 0;
	    cudaDeviceProp deviceProp;

	    cudaGetDeviceCount( &device_count );
	    // Find the best major SM Architecture GPU device
	    while ( current_device < device_count ) {
		    cudaGetDeviceProperties( &deviceProp, current_device );
		    if (deviceProp.major > 0 && deviceProp.major < 9999) {
			    best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		    }
		    current_device++;
	    }

        // Find the best CUDA capable GPU device
        current_device = 0;
        while( current_device < device_count ) {
           cudaGetDeviceProperties( &deviceProp, current_device );
           if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
               sm_per_multiproc = 1;
		   } else {
               sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
           }

           int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
           if( compute_perf  > max_compute_perf ) {
               // If we find GPU with SM major > 2, search only these
               if ( best_SM_arch > 2 ) {
                   // If our device==dest_SM_arch, choose this, or else pass
                   if (deviceProp.major == best_SM_arch) {	
                       max_compute_perf  = compute_perf;
                       max_perf_device   = current_device;
                   }
               } else {
                   max_compute_perf  = compute_perf;
                   max_perf_device   = current_device;
               }
           }
           ++current_device;
	    }
	    return max_perf_device;
    }

    // Initialization code to find the best CUDA Device
    int findCudaDevice(int argc, const char **argv)
    {
        cudaDeviceProp deviceProp;
        int devID = 0;
        // If the command-line has a device number specified, use it
        if (checkCmdLineFlag(argc, argv, "device")) {
            devID = getCmdLineArgumentInt(argc, argv, "device=");
            if (devID < 0) {
                printf("Invalid command line parameters\n");
                exit(-1);
            } else {
                devID = gpuDeviceInit(devID);
                if (devID < 0) {
                   printf("exiting...\n");
                   shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
                   exit(-1);
                }
            }
        } else {
            // Otherwise pick the device with highest Gflops/s
            devID = gpuGetMaxGflopsDeviceId();
            checkCudaErrors( cudaSetDevice( devID ) );
            checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
            printf("> Using CUDA device [%d]: %s\n", devID, deviceProp.name);
        }
        return devID;
    }
// end of CUDA Helper Functions

void AutoQATest()
{
    if (g_CheckRender && g_CheckRender->IsQAReadback()) {
        char temp[256];
        sprintf(temp, "%s <%s>", "[AutoTest]:", filterMode[g_Kernel]);  
	    glutSetWindowTitle(temp);

        g_Kernel++;

        if (g_Kernel > 3) {
            g_Kernel = 0;
            printf("Summary: %d errors!\n", g_TotalErrors);
            printf("\tTEST: %s\n", (g_TotalErrors==0) ? "OK" : "FAILURE");
            exit(0);
        }
    }
}


void computeFPS()
{
    frameCount++;
    fpsCount++;
    if (fpsCount == fpsLimit-1) {
        g_Verify = true;
    }
    if (fpsCount == fpsLimit) {
        char fps[256];
   		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "%s <%s>: %3.1f fps", 
                ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "[AutoTest]:" : ""),
				filterMode[g_Kernel], ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        if (g_CheckRender && !g_CheckRender->IsQAReadback()) fpsLimit = (int)MAX(ifps, 1.f);

        sdkResetTimer(&timer);  
        AutoQATest();
    }
}

void runImageFilters(TColor *d_dst)
{
    switch(g_Kernel){
        case 0:
            cuda_Copy(d_dst, imageW, imageH);
        break;
        case 1:
            if(!g_Diag)
                cuda_KNN(d_dst, imageW, imageH, 1.0f / (knnNoise * knnNoise), lerpC);
            else
                cuda_KNNdiag(d_dst, imageW, imageH, 1.0f / (knnNoise * knnNoise), lerpC);
        break;
        case 2:
            if(!g_Diag)
                cuda_NLM(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC); 
            else
                cuda_NLMdiag(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC);
        break;
        case 3:
            if(!g_Diag)
                cuda_NLM2(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC); 
            else
                cuda_NLM2diag(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC);
        break;
    }
    getLastCudaError("Filtering kernel execution failed.\n");
}


void displayFunc(void){
	sdkStartTimer(&timer);
    TColor *d_dst = NULL;
	size_t num_bytes;

    if(frameCounter++ == 0) sdkResetTimer(&timer);
    // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_dst, gl_PBO));
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	getLastCudaError("cudaGraphicsMapResources failed");
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_dst, &num_bytes, cuda_pbo_resource));
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

    checkCudaErrors( CUDA_Bind2TextureArray()                      );

    runImageFilters(d_dst);

    checkCudaErrors( CUDA_UnbindTexture()     );
    // DEPRECATED: checkCudaErrors(cudaGLUnmapBufferObject(gl_PBO));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	if (g_bFBODisplay) {
		g_FrameBufferObject->bindRenderPath();
	}

    // Common display code path
	{
        glClear(GL_COLOR_BUFFER_BIT);

        glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0) );
        glBegin(GL_TRIANGLES);
            glTexCoord2f(0, 0); glVertex2f(-1, -1);
            glTexCoord2f(2, 0); glVertex2f(+3, -1);
            glTexCoord2f(0, 2); glVertex2f(-1, +3);
        glEnd();
        glFinish();
    }

	if (g_bFBODisplay) {
		g_FrameBufferObject->unbindRenderPath();
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    if (g_CheckRender && g_CheckRender->IsQAReadback() && g_Verify) {
        printf("> (Frame %d) readback BackBuffer\n", frameCount);
        if (g_bFBODisplay) {
            g_CheckRender->readback( imageW, imageH, g_FrameBufferObject->getFbo() );
        } else {
            g_CheckRender->readback( imageW, imageH );
        }
        g_CheckRender->savePPM ( sOriginal[g_Kernel], true, NULL );
        if (!g_CheckRender->PPMvsPPM(sOriginal[g_Kernel], sReference[g_Kernel], MAX_EPSILON_ERROR, 0.15f)) {
            g_TotalErrors++;
        }
        g_Verify = false;
    }

    if(frameCounter == frameN){
        frameCounter = 0;
        if(g_FPS){
            printf("FPS: %3.1f\n", frameN / (sdkGetTimerValue(&timer) * 0.001) );
            g_FPS = false;
        }
    }

	glutSwapBuffers();

	sdkStopTimer(&timer);
	computeFPS();
}

void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}


void shutDown(unsigned char k, int /*x*/, int /*y*/)
{
    switch (k){
        case '\033':
        case 'q':
        case 'Q':
            printf("Shutting down...\n");

            sdkStopTimer(&timer);
            sdkDeleteTimer(&timer);

            checkCudaErrors( CUDA_FreeArray() );
            free(h_Src);

            shrQAFinishExit(*pArgc, (const char **)pArgv, QA_PASSED);
            exit(0);
        break;

        case '1':
            printf("Passthrough.\n");
            g_Kernel = 0;
        break;

        case '2':
            printf("KNN method \n");
            g_Kernel = 1;
        break;

        case '3':
            printf("NLM method\n");
            g_Kernel = 2;
        break;

        case '4':
            printf("Quick NLM(NLM2) method\n");
            g_Kernel = 3;
        break;

        case ' ':
            printf(g_Diag ? "LERP highlighting mode.\n" : "Normal mode.\n");
            g_Diag = !g_Diag;
        break;

        case 'n':
            printf("Decrease noise level.\n");
            knnNoise -= noiseStep;
            nlmNoise -= noiseStep;
        break;

        case 'N':
            printf("Increase noise level.\n");
            knnNoise += noiseStep;
            nlmNoise += noiseStep;
        break;

        case 'l':
            printf("Decrease LERP quotent.\n");
            lerpC = MAX(lerpC - lerpStep, 0.0f);
        break;

        case 'L':
            printf("Increase LERP quotent.\n");
            lerpC = MIN(lerpC + lerpStep, 1.0f);
        break;

        case 'f' : case 'F':
            g_FPS = true;
        break;

        case '?':
            printf("lerpC = %5.5f\n", lerpC);
            printf("knnNoise = %5.5f\n", knnNoise);
            printf("nlmNoise = %5.5f\n", nlmNoise);
        break;
    }
}


int initGL( int *argc, char **argv )
{
    printf("Initializing GLUT...\n");
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(imageW, imageH);
    glutInitWindowPosition(512 - imageW / 2, 384 - imageH / 2);
    glutCreateWindow(argv[0]);
    printf("OpenGL window created.\n");

    glewInit();
    printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
	if (g_bFBODisplay) {
        if (!glewIsSupported( "GL_VERSION_2_0 GL_ARB_fragment_program GL_EXT_framebuffer_object" )) {
            fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
            fprintf(stderr, "This sample requires:\n");
            fprintf(stderr, "  OpenGL version 2.0\n");
            fprintf(stderr, "  GL_ARB_fragment_program\n");
            fprintf(stderr, "  GL_EXT_framebuffer_object\n");
            exit(-1);
        }
	} else {
		if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
			fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
			fprintf(stderr, "This sample requires:\n");
			fprintf(stderr, "  OpenGL version 1.5\n");
			fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
			fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
            fflush(stderr);
            return false;
		}
	}

    return 0;
}

// shader for displaying floating-point texture
static const char *shader_code = 
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
    if (error_pos != -1) {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }
    return program_id;
}

void initOpenGLBuffers()
{
    printf("Creating GL texture...\n");
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &gl_Tex);
        glBindTexture(GL_TEXTURE_2D, gl_Tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageW, imageH, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
        glGenBuffers(1, &gl_PBO);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageW * imageH * 4, h_Src, GL_STREAM_COPY);
        //While a PBO is registered to CUDA, it can't be used 
        //as the destination for OpenGL drawing calls.
        //But in our particular case OpenGL is only used 
        //to display the content of the PBO, specified by CUDA kernels,
        //so we need to register/unregister it only once.
	// DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(gl_PBO) );
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, 
					       cudaGraphicsMapFlagsWriteDiscard));
    GLenum gl_error = glGetError();
	if (gl_error != GL_NO_ERROR) 
	{
#ifdef _WIN32
		char tmpStr[512];
		// NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
		// when the user double clicks on the error line in the Output pane. Like any compile error.
		sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", __FILE__, __LINE__, gluErrorString(gl_error));
		OutputDebugString(tmpStr);
#endif
		fprintf(stderr, "GL Error in file '%s' in line %d :\n", __FILE__, __LINE__);
		fprintf(stderr, "%s\n", gluErrorString(gl_error));
        shrQAFinishExit(*pArgc, (const char **)pArgv, QA_FAILED);
		exit(0);
	}

    printf("PBO created.\n");

    // load shader program
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

    if (g_FrameBufferObject) {
		delete g_FrameBufferObject; g_FrameBufferObject = NULL;
	}
	if (g_bFBODisplay) {
		g_FrameBufferObject = new CFrameBufferObject(imageW, imageH, 32, false, GL_TEXTURE_2D);
	}
}


void cleanup()
{
    sdkDeleteTimer( &timer );

    glDeleteProgramsARB(1, &shader);

    if (g_CheckRender) {
        delete g_CheckRender; g_CheckRender = NULL;
    }
    if (g_FrameBufferObject) {
        delete g_FrameBufferObject; g_FrameBufferObject = NULL;
    }
}

void runAutoTest(int argc, char **argv)
{
    printf("[%s] - (automated testing w/ readback)\n", sSDKsample);

	int devID = findCudaDevice(argc, (const char **)argv);

    // First load the image, so we know what the size of the image (imageW and imageH)
    printf("Allocating host and CUDA memory and loading image file...\n");
    const char *image_path = sdkFindFilePath("portrait_noise.bmp", argv[0]);
    if (image_path == NULL) {
       printf( "imageDenoisingGL was unable to find and load image file <portrait_noise.bmp>.\nExiting...\n");
       shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
    }
    LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
    printf("Data init done.\n");

    checkCudaErrors( CUDA_MallocArray(&h_Src, imageW, imageH) );

    g_CheckRender       = new CheckBackBuffer(imageW, imageH, sizeof(TColor), false);
    g_CheckRender->setExecPath(argv[0]);

    TColor *d_dst = NULL;
    checkCudaErrors( cudaMalloc( (void **)&d_dst, imageW*imageH*sizeof(TColor)) );

    while (g_Kernel <= 3) {
        printf("[AutoTest]: %s <%s>\n", sSDKsample, filterMode[g_Kernel]);
        checkCudaErrors( CUDA_Bind2TextureArray()                      );
        runImageFilters(d_dst);
        checkCudaErrors( CUDA_UnbindTexture()     );
        checkCudaErrors( cudaDeviceSynchronize() );
        cudaMemcpy(g_CheckRender->imageData(), d_dst, imageW*imageH*sizeof(TColor), cudaMemcpyDeviceToHost);
        g_CheckRender->savePPM(sOriginal[g_Kernel], true, NULL);

        if (!g_CheckRender->PPMvsPPM(sOriginal[g_Kernel], sReference[g_Kernel], MAX_EPSILON_ERROR, 0.15f)) {
            g_TotalErrors++;
        }
        g_Kernel++;
    }

    checkCudaErrors( CUDA_FreeArray() );
    free(h_Src);

    checkCudaErrors( cudaFree( d_dst ) );
    delete g_CheckRender;

	printf("\n[%s] -> Test Results: %d errors\n", sSDKsample, g_TotalErrors);

	cudaDeviceReset();
	shrQAFinishExit(argc, (const char **)argv, (!g_TotalErrors ? QA_PASSED : QA_FAILED));
}


int main(int argc, char **argv)
{
	pArgc = &argc;
	pArgv = argv;

    shrQAStart(argc, argv);

    if (argc > 1) {
        if (checkCmdLineFlag(argc, (const char **)argv, "qatest") ||
            checkCmdLineFlag(argc, (const char **)argv, "noprompt")) 
		{
            g_bQAReadback = true;
            fpsLimit = frameCheckNumber;
        }
        if (checkCmdLineFlag(argc, (const char **)argv, "glverify"))
		{
            g_bOpenGLQA = true;
            g_bFBODisplay = false;
            fpsLimit = frameCheckNumber;
        }
        if (checkCmdLineFlag(argc, (const char **)argv, "fbo")) {
            g_bFBODisplay = true;
            fpsLimit = frameCheckNumber;
        }
    }

    if (g_bQAReadback) {
        runAutoTest(argc, argv);
    } else {
        printf("[%s] ", sSDKsample);
        if (g_bFBODisplay) printf("[FBO Display] ");
        if (g_bOpenGLQA)   printf("[OpenGL Readback Comparisons] ");
        printf("\n");

		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		if ( checkCmdLineFlag(argc, (const char **)argv, "device")) {
			printf("[%s]\n", argv[0]);
			printf("   Does not explicitly support -device=n in OpenGL mode\n");
			printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
			printf(" > %s -device=n -qatest\n", argv[0]);
			printf("exiting...\n");
            exit(0);
		}

	    // First load the image, so we know what the size of the image (imageW and imageH)
        printf("Allocating host and CUDA memory and loading image file...\n");
        const char *image_path = sdkFindFilePath("portrait_noise.bmp", argv[0]);
        if (image_path == NULL) {
           printf( "imageDenoisingGL was unable to find and load image file <portrait_noise.bmp>.\nExiting...\n");
           shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
        }
        LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
        printf("Data init done.\n");

		// First initialize OpenGL context, so we can properly set the GL for CUDA.
		// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
		initGL( &argc, argv );
	    cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );

        checkCudaErrors( CUDA_MallocArray(&h_Src, imageW, imageH) );

        initOpenGLBuffers();

        // Creating the Auto-Validation Code
        if (g_bOpenGLQA) {
            if (g_bFBODisplay) {
                g_CheckRender = new CheckFBO(imageW, imageH, 4);
            } else {
                g_CheckRender = new CheckBackBuffer(imageW, imageH, 4);
            }
            g_CheckRender->setPixelFormat(GL_RGBA);
            g_CheckRender->setExecPath(argv[0]);
            g_CheckRender->EnableQAReadback(g_bOpenGLQA);
        }
    }

    printf("Starting GLUT main loop...\n");
    printf("Press [1] to view noisy image\n");
    printf("Press [2] to view image restored with knn filter\n");
    printf("Press [3] to view image restored with nlm filter\n");
    printf("Press [4] to view image restored with modified nlm filter\n");
    printf("Press [ ] to view smooth/edgy areas [RED/BLUE] Ct's\n");
    printf("Press [f] to print frame rate\n");
    printf("Press [?] to print Noise and Lerp Ct's\n");
    printf("Press [q] to exit\n");

    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(shutDown);

    sdkCreateTimer( &timer );
    sdkStartTimer( &timer );

    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    glutMainLoop();

    cudaDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
}
