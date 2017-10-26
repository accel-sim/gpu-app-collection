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
    Bicubic texture filtering sample
    sgreen 6/2008

    This sample demonstrates how to efficiently implement bicubic texture
    filtering in CUDA.

    Bicubic filtering is a higher order interpolation method that produces
    smoother results than bilinear interpolation:
    http://en.wikipedia.org/wiki/Bicubic

    It requires reading a 4 x 4 pixel neighbourhood rather than the
    2 x 2 area required by bilinear filtering.

    Current graphics hardware doesn't support bicubic filtering natively,
    but it is possible to compose a bicubic filter using just 4 bilinear
    lookups by offsetting the sample position within each texel and weighting
    the samples correctly. The only disadvantage to this method is that the
    hardware only maintains 9-bits of filtering precision within each texel.

    See "Fast Third-Order Texture Filtering", Sigg & Hadwiger, GPU Gems 2:
    http://developer.nvidia.com/object/gpu_gems_2_home.html

    v1.1 - updated to include the brute force method using 16 texture lookups.
    v1.2 - added Catmull-Rom interpolation
    
    Example performance results from GeForce 8800 GTS:
    Bilinear     - 5500 MPixels/sec
    Bicubic      - 1400 MPixels/sec
    Fast Bicubic - 2100 MPixels/sec
*/

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Helper functions
#include <rendercheck_gl.h> // CUDA utilities and system includes
#include <sdkHelper.h>      // CUDA SDK Helper functions
#include <shrQATest.h>      // Shared Library Test Functions

typedef unsigned int uint;
typedef unsigned char uchar;

#define USE_BUFFER_TEX 0
#ifndef MAX
#define MAX(a,b) ((a < b) ? b : a)
#endif
// Auto-Verification Code
const int frameCheckNumber = 4;
int       fpsCount = 0;        // FPS count for averaging
int       fpsLimit = 4;        // FPS limit for sampling
int       g_Index  = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
StopWatchInterface *timer = 0;
bool g_Verify      = false;

// CheckFBO/BackBuffer class objects
CheckRender        *g_CheckRender       = NULL;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY	  10 //ms

static const char *sSDKsample = "CUDA BicubicTexture"; 

// Define the files that are to be save and the reference images for validation
const char *sFilterMode[] =
{
    "Nearest",
    "Bilinear",
    "Bicubic",
    "Fast Bicubic",
    "Catmull-Rom",
    NULL
};

const char *sOriginal[] =
{
    "01_nearest.ppm",
    "02_bilinear.ppm",
    "03_bicubic.ppm",
    "04_fastbicubic.ppm",
	"05_catmull-rom.ppm",
    NULL
};

const char *sReference[] =
{
    "01_nearest.ppm",
    "02_bilinear.ppm",
    "03_bicubic.ppm",
    "04_fastbicubic.ppm",
    "05_catmull-rom.ppm",
    NULL
};

const char *imageFilename = "lena_bw.pgm";

uint width = 512, height = 512;
uint imageWidth, imageHeight;
dim3 blockSize(16, 16);
dim3 gridSize(width / blockSize.x, height / blockSize.y);

enum Mode {
	MODE_NEAREST, 
	MODE_BILINEAR, 
	MODE_BICUBIC, 
	MODE_FAST_BICUBIC, 
	MODE_CATMULL_ROM 
};

int mode = MODE_FAST_BICUBIC;

bool drawCurves = false;

GLuint pbo        = 0;          // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
GLuint displayTex = 0;
GLuint bufferTex  = 0;
GLuint fprog;                   // fragment program (shader)

float tx = 9.0f, ty = 10.0f;    // image translation
float scale = 1.0f / 16.0f;     // image scale
float cx, cy;                   // image centre

void display();
void displayImage();
void initGLBuffers();
void runBenchmark(int iterations);
void cleanup();

#define GL_TEXTURE_TYPE GL_TEXTURE_RECTANGLE_ARB
//#define GL_TEXTURE_TYPE GL_TEXTURE_2D

extern "C" void initGL(int *argc, char **argv);
extern "C" void loadImageData(int argc, char **argv);

extern "C" void initTexture(int imageWidth, int imageHeight, uchar *h_data);
extern "C" void freeTexture();
extern "C" void render( int width, int height, float tx, float ty, float scale, float cx, float cy, 
                        dim3 blockSize, dim3 gridSize, int mode, uchar4 *output);

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
            fprintf(stderr, ">> gpuGLDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
		    fprintf(stderr, "\n");
            return -dev;
        }
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
        if (deviceProp.major < 1) {
            fprintf(stderr, "Error: device does not support CUDA.\n");
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


// w0, w1, w2, and w3 are the four cubic B-spline basis functions
float bspline_w0(float a)
{
    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
}

float bspline_w1(float a)
{
    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
}

float bspline_w2(float a)
{
    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
}

__host__ __device__
float bspline_w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
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
        sprintf(fps, "%s %s <%s>: %3.1f fps", "", sSDKsample, sFilterMode[mode], ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 

        sdkResetTimer(&timer);  
    }
}

void plotCurve(float (*func)(float))
{
    const int steps = 100;
    glBegin(GL_LINE_STRIP);
    for(int i=0; i<steps; i++) {
        float x = i / (float) (steps-1);
        glVertex2f(x, func(x));
    }
    glEnd();
}

// display results using OpenGL (called by GLUT)
void display()
{
    sdkStartTimer(&timer);  

    // map PBO to get CUDA device pointer
    uchar4 *d_output;
    //DEPRECATED: checkCudaErrors( cudaGLMapBufferObject((void**)&d_output, pbo) );
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes; 
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  
						       cuda_pbo_resource));
    render( imageWidth, imageHeight, tx, ty, scale, cx, cy, 
            blockSize, gridSize, mode, d_output );

    // DEPRECATED: checkCudaErrors(cudaGLUnmapBufferObject(pbo));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    // Common diplay path
    {
	// display results
	glClear(GL_COLOR_BUFFER_BIT);
	
#if USE_BUFFER_TEX
	// display using buffer texture
	glBindTexture(GL_TEXTURE_BUFFER_EXT, bufferTex);
	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, fprog);
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glProgramLocalParameterI4iNV(GL_FRAGMENT_PROGRAM_ARB, 0, width, 0, 0, 0);
#else
	// download image from PBO to OpenGL texture
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture  (GL_TEXTURE_TYPE, displayTex);
	glPixelStorei  (GL_UNPACK_ALIGNMENT, 1);
	glTexSubImage2D(GL_TEXTURE_TYPE,
			0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, 0);
	glEnable(GL_TEXTURE_TYPE);
#endif
	
	// draw textured quad
	glDisable(GL_DEPTH_TEST);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f          , (GLfloat)height);  glVertex2f(0.0f, 0.0f);
	glTexCoord2f((GLfloat)width, (GLfloat)height);  glVertex2f(1.0f, 0.0f);
	glTexCoord2f((GLfloat)width, 0.0f           );  glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f          , 0.0f           );  glVertex2f(0.0f, 1.0f);
	glEnd();
	glDisable(GL_TEXTURE_TYPE);
	glDisable(GL_FRAGMENT_PROGRAM_ARB);
	
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	
	if (drawCurves) {
	    // draw spline curves
	    glPushMatrix();
	    glScalef(0.25, 0.25, 1.0);
	    
	    glTranslatef(0.0, 2.0, 0.0);
	    glColor3f(1.0, 0.0, 0.0);
	    plotCurve(bspline_w3);
	    
	    glTranslatef(1.0, 0.0, 0.0);
	    glColor3f(0.0, 1.0, 0.0);
	    plotCurve(bspline_w2);
	    
	    glTranslatef(1.0, 0.0, 0.0);
	    glColor3f(0.0, 0.0, 1.0);
	    plotCurve(bspline_w1);
	    
	    glTranslatef(1.0, 0.0, 0.0);
	    glColor3f(1.0, 0.0, 1.0);
	    plotCurve(bspline_w0);
	    
	    glPopMatrix();
	    glColor3f(1.0, 1.0, 1.0);
	}
    }
    
    glutSwapBuffers();
    glutReportErrors();
    
    sdkStopTimer(&timer);  
    
    computeFPS();
}

// GLUT callback functions
void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}


void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch(key) {
        case 27:
            cleanup();
            exit(0);
            break;
        case '1':
            mode = MODE_NEAREST;
			break;
        case '2':
            mode = MODE_BILINEAR;
            break;
        case '3':
            mode = MODE_BICUBIC;
            break;
        case '4':
            mode = MODE_FAST_BICUBIC;
            break;
        case '5':
            mode = MODE_CATMULL_ROM;
            break;
        case '=':
        case '+':
            scale *= 0.5f;
            break;
        case '-':
            scale *= 2.0f;
            break;
        case 'r':
            scale = 1.0f;
            tx = ty = 0.0f;
            break;
        case 'd':
            printf("%f, %f, %f\n", tx, ty, scale);
			break;
        case 'b':
            runBenchmark(500);
            break;
        case 'c':
            drawCurves ^= 1;
            break;
        default:
            break;
    }
	if (key >= '1' && key <= '5') {
		printf("> FilterMode[%d] = %s\n", mode+1, sFilterMode[mode]);
	}
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState & 1) {
        // left = translate
        tx -= dx*scale;
        ty -= dy*scale;
    }
    else if (buttonState & 2) {
        // middle = zoom
        scale -= dy / 1000.0f;
    }

    ox = x; oy = y;
}

void reshape(int x, int y)
{
    width = x; height = y;
    imageWidth = width; imageHeight = height;

    initGLBuffers();

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

void cleanup()
{
    freeTexture();
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));	

    glDeleteBuffersARB(1, &pbo);

#if USE_BUFFER_TEX
    glDeleteTextures(1, &bufferTex);
    glDeleteProgramsARB(1, &fprog);
#else
    glDeleteTextures(1, &displayTex);
#endif
}

int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void initGLBuffers()
{
    if (pbo) {
        // delete old buffer
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
        glDeleteBuffersARB(1, &pbo);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(uchar4), 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, 
					       cudaGraphicsMapFlagsWriteDiscard));

#if USE_BUFFER_TEX
    // create buffer texture, attach to pbo
    if (bufferTex) {
        glDeleteTextures(1, &bufferTex);
    }
    glGenTextures(1, &bufferTex);
    glBindTexture(GL_TEXTURE_BUFFER_EXT, bufferTex);
    glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA8, pbo);
    glBindTexture(GL_TEXTURE_BUFFER_EXT, 0);
#else
    // create texture for display
    if (displayTex) {
        glDeleteTextures(1, &displayTex);
    }
    glGenTextures(1, &displayTex);
    glBindTexture  (GL_TEXTURE_TYPE, displayTex);
    glTexImage2D   (GL_TEXTURE_TYPE, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture  (GL_TEXTURE_TYPE, 0);
#endif

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}

void mainMenu(int i)
{
    keyboard(i, 0, 0);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Nearest      [1]", '1');
    glutAddMenuEntry("Bilinear     [2]", '2');
    glutAddMenuEntry("Bicubic      [3]", '3');
    glutAddMenuEntry("Fast Bicubic [4]", '4');
    glutAddMenuEntry("Catmull-Rom  [5]", '5');
    glutAddMenuEntry("Zoom in      [=]", '=');
    glutAddMenuEntry("Zoom out     [-]", '-');
    glutAddMenuEntry("Benchmark    [b]", 'b');
    glutAddMenuEntry("DrawCurves   [c]", 'c');
    glutAddMenuEntry("Quit       [esc]", 27);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void runBenchmark(int iterations)
{
    printf("[%s] (Benchmark Mode)\n", sSDKsample);

    sdkCreateTimer(&timer);

    uchar4 *d_output;
    // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_output, pbo));
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes; 
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  
						       cuda_pbo_resource));

    sdkStartTimer(&timer);  
    for (int i = 0; i < iterations; ++i)
    {
        render( imageWidth, imageHeight, tx, ty, scale, cx, cy, 
                blockSize, gridSize, mode, d_output);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);  
    float time = sdkGetTimerValue(&timer) / (float) iterations;

    //DEPRECATED: checkCudaErrors(cudaGLUnmapBufferObject(pbo));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    printf("time: %0.3f ms, %f Mpixels/sec\n", time, (width*height / (time * 0.001f)) / 1e6);    
}


void runAutoTest(int argc, char **argv)
{
    cudaDeviceProp deviceProps;

    int devID = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));

	printf("[%s] (automated testing w/ readback)\n", sSDKsample);
	printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);

	loadImageData(argc, argv);

    g_CheckRender       = new CheckBackBuffer(imageWidth, imageHeight, 4, false);
    g_CheckRender->setExecPath(argv[0]);

    uchar4 *d_output;
    checkCudaErrors( cudaMalloc( (void **)&d_output, imageWidth*imageHeight*4 ) );

    for (int mode=0; mode <= MODE_CATMULL_ROM; mode++) {
        printf("AutoTest: %s <%s>\n", sSDKsample, sFilterMode[mode] );

        render( imageWidth, imageHeight, 
                tx, ty, scale, cx, cy, 
                blockSize, gridSize, mode, d_output);

        // check if kernel execution generated an error
        getLastCudaError("Error: render (bicubicTexture) Kernel execution FAILED");
        checkCudaErrors( cudaDeviceSynchronize() );

        cudaMemcpy(g_CheckRender->imageData(), d_output, imageWidth*imageHeight*4, cudaMemcpyDeviceToHost);

        g_CheckRender->savePPM(sOriginal[mode], false, NULL);

        if (!g_CheckRender->PPMvsPPM(sOriginal[mode], sReference[mode], MAX_EPSILON_ERROR, 0.15f)) {
            g_TotalErrors++;
        }
	}

    checkCudaErrors( cudaFree( d_output ) );
    delete g_CheckRender;

    cudaDeviceReset();

    shrQAFinishExit(argc, (const char **)argv, (g_TotalErrors == 0 ? QA_PASSED : QA_FAILED));
}


#if USE_BUFFER_TEX
// fragment program for reading from buffer texture
static const char *shaderCode = 
"!!NVfp4.0\n"
"INT PARAM width = program.local[0];\n"
"INT TEMP index;\n"
"FLR.S index, fragment.texcoord;\n"
"MAD.S index.x, index.y, width, index.x;\n" // compute 1D index from 2D coords
"TXF result.color, index.x, texture[0], BUFFER;\n"
"END";
#endif

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

void initialize(int argc, char **argv)
{
    printf("[%s] (OpenGL Mode)\n", sSDKsample);

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    initGL( &argc, argv );

    int devID;
    cudaDeviceProp deviceProps;

	if( checkCmdLineFlag(argc, (const char**)argv, "device") ) {
        devID = gpuGLDeviceInit(argc, argv);
        if (devID < 0) {
            printf("exiting...\n");
            shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
        }
	} else {
        devID = gpuGetMaxGflopsDeviceId();
        cudaGLSetGLDevice( devID );
	}
    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
	printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);

    // Create the timer (for fps measurement)
    sdkCreateTimer( &timer );

    // load image from disk
    loadImageData(argc, argv);

    printf("\n"
        "\tControls\n"
		"\t=/- : Zoom in/out\n"
		"\tb   : Run Benchmark mode\n"
		"\tc   : Draw Bicubic Spline Curve\n"
		"\t[esc] - Quit\n\n"

		"\tPress number keys to change filtering mode:\n\n"
		"\t1 : nearest filtering\n"
		"\t2 : bilinear filtering\n"
		"\t3 : bicubic filtering\n"
		"\t4 : fast bicubic filtering\n"
		"\t5 : Catmull-Rom filtering\n\n"
        );

    initGLBuffers();

#if USE_BUFFER_TEX
    fprog = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shaderCode);
    if (!fprog) exit(0);
#endif
}

void initGL( int *argc, char **argv )
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA bicubic texture filtering");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    initMenus();

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 "
                         "GL_ARB_pixel_buffer_object "
                         )) 
    {
        fprintf(stderr, "Required OpenGL extensions are missing.");
        exit(-1);
    }
#if USE_BUFFER_TEX
    if (!glewIsSupported("GL_EXT_texture_buffer_object")) {
       fprintf(stderr, "OpenGL extension: GL_EXT_texture_buffer_object missing.\n");
       exit(-1);
    }
    if (!glewIsSupported("GL_NV_gpu_program4")) {
       fprintf(stderr, "OpenGL extension: GL_NV_gpu_program4 missing.\n");
       exit(-1);
    }
#endif
}

void loadImageData(int argc, char **argv)
{
    // load image from disk
    uchar* h_data = NULL;
    char* imagePath = NULL;

    if (argc >= 1)
	if ( (imagePath = sdkFindFilePath(imageFilename, argv[0])) == NULL) {
        printf("bicubicTexture loadImageData() could not find <%s>\nExiting...\n", imageFilename);
        shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
    }

    sdkLoadPGM<unsigned char>(imagePath, &h_data, &imageWidth, &imageHeight);

    printf("Loaded '%s', %d x %d pixels\n", imageFilename, imageWidth, imageHeight);

    cx = imageWidth * 0.5f;
    cy = imageHeight * 0.5f;

    // initialize texture
    initTexture(imageWidth, imageHeight, h_data);
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

	// parse arguments
    char *filename;
    if (argc > 1) {
        if (getCmdLineArgumentString( argc, (const char**) argv, "file", &filename)) {
            imageFilename = filename;
        }
        if (checkCmdLineFlag(argc, (const char **)argv, "qatest") ||
            checkCmdLineFlag(argc, (const char **)argv, "noprompt")) 
        {
            fpsLimit = frameCheckNumber;
        }
    }


    if ( true || checkCmdLineFlag(argc, (const char **)argv, "qatest") ||
		checkCmdLineFlag(argc, (const char **)argv, "noprompt")) 
	{
		// Running CUDA kernel (bicubicFiltering) without visualization (QA Testing/Verification)
        runAutoTest(argc, argv);
	} 
	else 
	{
		// This runs the CUDA kernel (bicubicFiltering) + OpenGL visualization
		initialize(argc, argv);
		glutMainLoop();
		sdkDeleteTimer( &timer );
		cudaDeviceReset();
		shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
	}

    return 0;
}
