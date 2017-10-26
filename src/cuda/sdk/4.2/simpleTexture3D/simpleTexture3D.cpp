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
    3D texture sample

    This sample loads a 3D volume from disk and displays slices through it
    using 3D texture lookups.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <rendercheck_gl.h>
#include <sdkHelper.h>    // includes cuda.h and cuda_runtime_api.h
#include <shrQATest.h>    // standard utility and system includes
#include <vector_types.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

#define MAX_EPSILON_ERROR 5.0f
#define THRESHOLD         0.15f

const char *sSDKsample = "simpleTexture3D";

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "simpleTex3D.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_simpleTex3D.ppm",
    NULL
};

const char *volumeFilename = "Bucky.raw";
const cudaExtent volumeSize = make_cudaExtent(32, 32, 32);

const uint width = 512, height = 512;
const dim3 blockSize(16, 16, 1);
const dim3 gridSize(width / blockSize.x, height / blockSize.y);

float w = 0.5;  // texture coordinate in z

GLuint pbo;     // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

bool linearFiltering = true;
bool animate = true;

StopWatchInterface *timer;

uint *d_output = NULL;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_Verify = false;
bool g_bQAReadback = false;
bool g_bOpenGLQA   = false;

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void initCuda(const uchar *h_volume, cudaExtent volumeSize);
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float w);

void loadVolumeData(char *exec_path);

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

        if (deviceCount == 0)
        {
            fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
            exit(-1);
        }

        if (devID < 0)
           devID = 0;
            
        if (devID > deviceCount-1)
        {
            fprintf(stderr, "\n");
            fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
            fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
            fprintf(stderr, "\n");
            return -devID;
        }

        cudaDeviceProp deviceProp;
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );

        if (deviceProp.major < 1)
        {
            fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
            exit(-1);                                                  
        }
        
        checkCudaErrors( cudaSetDevice(devID) );
        printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

        return devID;
    }

    // This function returns the best GPU (with maximum GFLOPS)
    int gpuGetMaxGflopsDeviceId()
    {
        int current_device     = 0, sm_per_multiproc  = 0;
        int max_compute_perf   = 0, max_perf_device   = 0;
        int device_count       = 0, best_SM_arch      = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceCount( &device_count );
        
        // Find the best major SM Architecture GPU device
        while (current_device < device_count)
        {
            cudaGetDeviceProperties( &deviceProp, current_device );
            if (deviceProp.major > 0 && deviceProp.major < 9999)
            {
                best_SM_arch = MAX(best_SM_arch, deviceProp.major);
            }
            current_device++;
        }

        // Find the best CUDA capable GPU device
        current_device = 0;
        while( current_device < device_count )
        {
            cudaGetDeviceProperties( &deviceProp, current_device );
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
            }
            
            int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
            
        if( compute_perf  > max_compute_perf )
        {
                // If we find GPU with SM major > 2, search only these
                if ( best_SM_arch > 2 )
                {
                    // If our device==dest_SM_arch, choose this, or else pass
                    if (deviceProp.major == best_SM_arch)
                    {
                        max_compute_perf  = compute_perf;
                        max_perf_device   = current_device;
                     }
                }
                else
                {
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
        if (checkCmdLineFlag(argc, argv, "device"))
        {
            devID = getCmdLineArgumentInt(argc, argv, "device=");
            if (devID < 0)
            {
                printf("Invalid command line parameter\n ");
                exit(-1);
            }
            else
            {
                devID = gpuDeviceInit(devID);
                if (devID < 0)
                {
                    printf("exiting...\n");
                    shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
                    exit(-1);
                }
            }
        }
        else
        {
            // Otherwise pick the device with highest Gflops/s
            devID = gpuGetMaxGflopsDeviceId();
            checkCudaErrors( cudaSetDevice( devID ) );
            checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
            printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
        }
        return devID;
    }
   
    inline int gpuGLDeviceInit(int ARGC, char **ARGV)
    {
        int deviceCount;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
            exit(-1);
        }
        int dev = 0;
        dev = getCmdLineArgumentInt(ARGC, (const char **) ARGV, "device=");
        if (dev < 0)
            dev = 0;
        if (dev > deviceCount-1) {
		    fprintf(stderr, "\n");
		    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
            fprintf(stderr, ">> cutilDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
		    fprintf(stderr, "\n");
            return -dev;
        }
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
        if (deviceProp.major < 1) {
            fprintf(stderr, "cutil error: device does not support CUDA.\n");
            exit(-1);                                                  \
        }
        if (checkCmdLineFlag(ARGC, (const char **) ARGV, "quiet") == false)
            fprintf(stderr, "Using device %d: %s\n", dev, deviceProp.name);

        checkCudaErrors(cudaGLSetGLDevice(dev));
        return dev;
    }

    // This function will pick the best CUDA device available with OpenGL interop
    inline int findCudaGLDevice(int argc, char **argv)
    {
	    int devID = 0;
        // If the command-line has a device number specified, use it
        if( checkCmdLineFlag(argc, (const char**)argv, "device") ) {
		    devID = gpuGLDeviceInit(argc, argv);
		    if (devID < 0) {
		       printf("exiting...\n");
		       cudaDeviceReset();
		       exit(0);
		    }
        } else {
            // Otherwise pick the device with highest Gflops/s
		    devID = gpuGetMaxGflopsDeviceId();
            cudaGLSetGLDevice( devID );
        }
	    return devID;
    }

    ////////////////////////////////////////////////////////////////////////////
    //! Check for OpenGL error
    //! @return CUTTrue if no GL error has been encountered, otherwise 0
    //! @param file  __FILE__ macro
    //! @param line  __LINE__ macro
    //! @note The GL error is listed on stderr
    //! @note This function should be used via the CHECK_ERROR_GL() macro
    ////////////////////////////////////////////////////////////////////////////
    inline bool
    sdkCheckErrorGL( const char* file, const int line) 
    {
	    bool ret_val = true;

	    // check for error
	    GLenum gl_error = glGetError();
	    if (gl_error != GL_NO_ERROR) 
	    {
    #ifdef _WIN32
		    char tmpStr[512];
		    // NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
		    // when the user double clicks on the error line in the Output pane. Like any compile error.
		    sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", file, line, gluErrorString(gl_error));
		    OutputDebugString(tmpStr);
    #endif
		    fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
		    fprintf(stderr, "%s\n", gluErrorString(gl_error));
		    ret_val = false;
	    }
	    return ret_val;
    }

    #define SDK_CHECK_ERROR_GL()                                              \
	    if( false == sdkCheckErrorGL( __FILE__, __LINE__)) {                  \
	        exit(EXIT_FAILURE);                                               \
	    }
// end of CUDA Helper Functions

void AutoQATest()
{
    if (g_CheckRender && g_CheckRender->IsQAReadback()) {
        char temp[256];
        sprintf(temp, "AutoTest: %s", sSDKsample);
        glutSetWindowTitle(temp);
        shrQAFinishExit2(true, *pArgc, (const char **)pArgv, QA_PASSED);
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
        sprintf(fps, "%s %s: %3.1f fps", sSDKsample, 
                ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        if (g_CheckRender && !g_CheckRender->IsQAReadback()) fpsLimit = (int)MAX(ifps, 1.f);

        sdkResetTimer(&timer);  

        AutoQATest();
    }
}


// render image using CUDA
void render()
{
    // map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes; 
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // call CUDA kernel, writing results to PBO
    render_kernel(gridSize, blockSize, d_output, width, height, w);

    getLastCudaError("render_kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
    sdkStartTimer(&timer);  

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    if (g_CheckRender && g_CheckRender->IsQAReadback() && g_Verify) {
        // readback for QA testing
        printf("> (Frame %d) Readback BackBuffer\n", frameCount);
        g_CheckRender->readback( width, height );
        g_CheckRender->savePPM(sOriginal[g_Index], true, NULL);
        if (!g_CheckRender->PPMvsPPM(sOriginal[g_Index], sReference[g_Index], MAX_EPSILON_ERROR, THRESHOLD)) {
            g_TotalErrors++;
        }
        g_Verify = false;
    }

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);  
    computeFPS();
}

void idle()
{
    if (animate) {
        w += 0.01f;
        glutPostRedisplay();
    }
}

void keyboard(unsigned char key, int x, int y)
{
    switch(key) {
        case 27:
            exit(0);
            break;
        case '=':
        case '+':
            w += 0.01f;
            break;
        case '-':
            w -= 0.01f;
            break;
        case 'f':
            linearFiltering = !linearFiltering;
            setTextureFilterMode(linearFiltering);
            break;
        case ' ':
            animate = !animate;
            break;
        default:
            break;
    }
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

void cleanup()
{
    sdkDeleteTimer( &timer);

    if (!g_bQAReadback) {
		// unregister this buffer object from CUDA C
		cudaGraphicsUnregisterResource(cuda_pbo_resource);

		glDeleteBuffersARB(1, &pbo);
    }

    if (g_CheckRender) {
        delete g_CheckRender; g_CheckRender = NULL;
    }
}

void initGLBuffers()
{
    // create pixel buffer object
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));	
}

// Load raw data from disk
uchar *loadRawFile(const char *filename, size_t size)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    uchar *data = (uchar *) malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    printf("Read '%s', %lu bytes\n", filename, read);

    return data;
}

void initGL( int *argc, char **argv )
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA 3D texture");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }
}

// General initialization call for CUDA Device
int chooseCudaDevice(int argc, char **argv, bool bUseOpenGL)
{
	int result = 0;
    if (bUseOpenGL) {
        result = findCudaGLDevice(argc, argv);
    } else {
        result = findCudaDevice(argc, (const char **)argv);
    }
	return result;
}

void runAutoTest( int argc, char **argv )
{
    g_CheckRender = new CheckBackBuffer(width, height, 4, false);
    g_CheckRender->setPixelFormat(GL_RGBA);
    g_CheckRender->setExecPath(argv[0]);
    g_CheckRender->EnableQAReadback(true);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
	chooseCudaDevice(argc, argv, false);

    loadVolumeData(argv[0]);

    checkCudaErrors( cudaMalloc((void **)&d_output, width*height*sizeof(GLubyte)*4) );

    // render the volumeData
    render_kernel(gridSize, blockSize, d_output, width, height, w);

    checkCudaErrors( cudaDeviceSynchronize() );
    getLastCudaError("render_kernel failed");

    checkCudaErrors( cudaMemcpy( g_CheckRender->imageData(), d_output, width*height*sizeof(GLubyte)*4, cudaMemcpyDeviceToHost) );
    g_CheckRender->dumpBin((void *)g_CheckRender->imageData(), width*height*sizeof(GLubyte)*4, "simpleTexture3D.bin");
    if (!g_CheckRender->compareBin2BinFloat("simpleTexture3D.bin", "ref_texture3D.bin", width*height*sizeof(GLubyte)*4, MAX_EPSILON_ERROR, THRESHOLD))
       g_TotalErrors++;

    checkCudaErrors( cudaFree(d_output) );
}


void loadVolumeData(char *exec_path)
{
    // load volume data
    const char* path = sdkFindFilePath(volumeFilename, exec_path);
    if (path == NULL) {
        fprintf(stderr, "Error unable to find 3D Volume file: '%s'\n", volumeFilename);
        shrQAFinishExit2(false, *pArgc, (const char **)pArgv, QA_FAILED);
    }

    size_t size = volumeSize.width*volumeSize.height*volumeSize.depth;
    uchar *h_volume = loadRawFile(path, size);

    initCuda(h_volume, volumeSize);
    sdkCreateTimer( &timer );

    free(h_volume);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	pArgc = &argc;
	pArgv = argv;

	shrQAStart(argc, argv);
    printf("[%s] ", sSDKsample);

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
            fpsLimit = frameCheckNumber;
        }
    }

    if (g_bQAReadback) printf("(Automated Testing)\n");
    if (g_bOpenGLQA)   printf("(OpenGL Readback)\n");

    if (g_bQAReadback) {
        runAutoTest(argc, argv);
        cleanup();
        cudaDeviceReset();
        shrQAFinishExit(argc, (const char **)argv, (g_TotalErrors > 0) ? QA_FAILED : QA_PASSED);
    } 
    else 
    {
        if (g_bOpenGLQA) {
            g_CheckRender = new CheckBackBuffer(width, height, 4);
            g_CheckRender->setPixelFormat(GL_RGBA);
            g_CheckRender->setExecPath(argv[0]);
            g_CheckRender->EnableQAReadback(true);
        }

        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        initGL(&argc, argv);

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
		chooseCudaDevice(argc, argv, true);

        // OpenGL buffers
        initGLBuffers();

        loadVolumeData(argv[0]);
    }

    printf("Press space to toggle animation\n"
           "Press '+' and '-' to change displayed slice\n");

    atexit(cleanup);

    glutMainLoop();

    cudaDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
}
