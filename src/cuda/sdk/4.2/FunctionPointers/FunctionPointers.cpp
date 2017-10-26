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

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_gl_inline.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h
#include <rendercheck_gl.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "FunctionPointers_kernels.h"

// Shared Library Test Functions
#include <shrQATest.h>


#define MIN_RUNTIME_VERSION 3010
#define MIN_COMPUTE_VERSION 0x20

//
// Cuda example code that implements the Sobel edge detection
// filter. This code works for 8-bit monochrome images.
//
// Use the '-' and '=' keys to change the scale factor.
//
// Other keys:
// I: display image
// T: display Sobel Edge Detection (computed solely with texture)
// S: display Sobel Edge Detection (computed with texture and shared memory)

void cleanup(void);
void initializeData(char *file, int argc, char **argv) ;

#define MAX_EPSILON_ERROR 5.0f

static char *sSDKsample = "CUDA Function Pointers (SobelFilter)"; 

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "lena_orig.pgm",
    "lena_shared.pgm",
    "lena_shared_tex.pgm",
    NULL
};

const char *sOriginal_ppm[] =
{
    "lena_orig.ppm",
    "lena_shared.ppm",
    "lena_shared_tex.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_orig.pgm",
    "ref_shared.pgm",
    "ref_shared_tex.pgm",
    NULL
};

const char *sReference_ppm[] =
{
    "ref_orig.ppm",
    "ref_shared.ppm",
    "ref_shared_tex.ppm",
    NULL
};

static int wWidth   = 512; // Window width
static int wHeight  = 512; // Window height
static int imWidth  = 0;   // Image width
static int imHeight = 0;   // Image height
static int blockOp = 0;
static int pointOp = 1;

// Code to handle Auto verification
const int frameCheckNumber = 4;
int fpsCount = 0;      // FPS count for averaging
int fpsLimit = 8;      // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool       g_Verify = false, g_AutoQuit = false;
unsigned int timer;
unsigned int g_Bpp;
unsigned int g_Index = 0;

bool g_bQAReadback = false;
bool g_bOpenGLQA   = false;
bool g_bFBODisplay = false;

int *pArgc = NULL;
char **pArgv = NULL;

// Display Data
static GLuint pbo_buffer = 0;  // Front and back CA buffers
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

static GLuint texid = 0;       // Texture for display
unsigned char * pixels = NULL; // Image pixel data on the host    
float imageScale = 1.f;        // Image exposure
enum SobelDisplayMode g_SobelDisplayMode;

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

bool bQuit          = false;

extern "C" void runAutoTest(int argc, char **argv);

#define OFFSET(i) ((char *)NULL + (i))
#define MAX(a,b) ((a > b) ? a : b)
#define REFRESH_DELAY	  10 //ms

void AutoQATest()
{
    if (g_CheckRender && g_CheckRender->IsQAReadback()) {
        char temp[256];
        g_SobelDisplayMode = (SobelDisplayMode)g_Index;
        sprintf(temp, "%s FunctionPointers [CUDA Edge Detection] (%s)", "AutoTest:", filterMode[g_SobelDisplayMode]);  
	    glutSetWindowTitle(temp);

        g_Index++;
        if (g_Index > 2) {
            g_Index = 0;
            printf("Summary: %d errors!\n", g_TotalErrors);
            printf("%s\n", (g_TotalErrors==0) ? "PASSED" : "FAILED");
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
        float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
        sprintf(fps, "%s FunctionPointers [CUDA Edge Detection] (%s): %3.1f fps", 
                ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""),
				filterMode[g_SobelDisplayMode], ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        if (g_CheckRender && !g_CheckRender->IsQAReadback()) fpsLimit = (int)MAX(ifps, 1.f);

        cutilCheckError(cutResetTimer(timer));  
        AutoQATest();
    }
}


// This is the normal display path
void display(void) 
{  
    cutilCheckError(cutStartTimer(timer));  

    // Sobel operation
    Pixel *data = NULL;

    // map PBO to get CUDA device pointer
	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes; 
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&data, &num_bytes,  
						       cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);
	
	sobelFilter(data, imWidth, imHeight, g_SobelDisplayMode, imageScale, blockOp, pointOp );
    cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, texid);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imWidth, imHeight, 
                   GL_LUMINANCE, GL_UNSIGNED_BYTE, OFFSET(0));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        
    glBegin(GL_QUADS);
    glVertex2f(0, 0); glTexCoord2f(0, 0);
    glVertex2f(0, 1); glTexCoord2f(1, 0);
    glVertex2f(1, 1); glTexCoord2f(1, 1);
    glVertex2f(1, 0); glTexCoord2f(0, 1);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);

    if (g_CheckRender && g_CheckRender->IsQAReadback() && g_Verify) {
        printf("> (Frame %d) readback BackBuffer\n", frameCount);
        g_CheckRender->readback( imWidth, imHeight );
        g_CheckRender->savePPM ( sOriginal_ppm[g_Index], true, NULL );
        if (!g_CheckRender->PPMvsPPM(sOriginal_ppm[g_Index], sReference_ppm[g_Index], MAX_EPSILON_ERROR, 0.15f)) {
            g_TotalErrors++;
        }
        g_Verify = false;
    }
    glutSwapBuffers();

    cutilCheckError(cutStopTimer(timer));  

    computeFPS();
}

void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void keyboard( unsigned char key, int /*x*/, int /*y*/) 
{
    char temp[256];

    switch( key) {
        case 27:
            exit (0);
	    break;
	case '-':
	    imageScale -= 0.1f;
            printf("brightness = %4.2f\n", imageScale);
	    break;
	case '=':
            imageScale += 0.1f;
            printf("brightness = %4.2f\n", imageScale);
	    break;
        case 'i': 
        case 'I':
            g_SobelDisplayMode = SOBELDISPLAY_IMAGE;
            sprintf(temp, "Function Pointers [CUDA Edge Detection] (%s)", filterMode[g_SobelDisplayMode]);  
             glutSetWindowTitle(temp);
	    break;
	case 's': 
        case 'S':
            g_SobelDisplayMode = SOBELDISPLAY_SOBELSHARED;
            sprintf(temp, "Function Pointers [CUDA Edge Detection] (%s)", filterMode[g_SobelDisplayMode]);  
            glutSetWindowTitle(temp);
            break;
        case 't': 
        case 'T':
            g_SobelDisplayMode = SOBELDISPLAY_SOBELTEX;
            sprintf(temp, "Function Pointers [CUDA Edge Detection] (%s)", filterMode[g_SobelDisplayMode]);  
            glutSetWindowTitle(temp);
            break;
	case 'b':
	case 'B':
            blockOp = (blockOp + 1)%LAST_BLOCK_FILTER;
            break;
	case 'p':
	case 'P':
            pointOp = (pointOp +1)%LAST_POINT_FILTER;
            break;
        default: break;

    }
}

void reshape(int x, int y) {
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, 0, 1); 
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void cleanup(void) {
	cudaGraphicsUnregisterResource(cuda_pbo_resource);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &pbo_buffer);
    glDeleteTextures(1, &texid);
    deleteTexture();

    if (g_CheckRender) {
        delete g_CheckRender; g_CheckRender = NULL;
    }

    cutilCheckError(cutDeleteTimer(timer));  
}

void initializeData(char *file, int argc, char **argv) {
    GLint bsize;
    unsigned int w, h;
    size_t file_length= strlen(file);

    if (!strcmp(&file[file_length-3], "pgm")) {
        if (cutLoadPGMub(file, &pixels, &w, &h) != CUTTrue) {
            printf("Failed to load image file: %s\n", file);
            exit(-1);
        }
        g_Bpp = 1;
    } else if (!strcmp(&file[file_length-3], "ppm")) {
        if (cutLoadPPM4ub(file, &pixels, &w, &h) != CUTTrue) {
            printf("Failed to load image file: %s\n", file);
            exit(-1);
        }
        g_Bpp = 4;
    } else {
        cutilDeviceReset();
        shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
    }
    imWidth = (int)w; imHeight = (int)h;
    setupTexture(imWidth, imHeight, pixels, g_Bpp);

    // copy function pointer tables to host side for later use
    setupFunctionTables();

    memset(pixels, 0x0, g_Bpp * sizeof(Pixel) * imWidth * imHeight);

    if (!g_bQAReadback) {
        // use OpenGL Path
        glGenBuffers(1, &pbo_buffer);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer); 
        glBufferData(GL_PIXEL_UNPACK_BUFFER, 
                        g_Bpp * sizeof(Pixel) * imWidth * imHeight, 
                        pixels, GL_STREAM_DRAW);  

        glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize); 
        if ((GLuint)bsize != (g_Bpp * sizeof(Pixel) * imWidth * imHeight)) {
            printf("Buffer object (%d) has incorrect size (%d).\n", (unsigned)pbo_buffer, (unsigned)bsize);
            cutilDeviceReset();
            shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
        }

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // register this buffer object with CUDA
        cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard));	

        glGenTextures(1, &texid);
        glBindTexture(GL_TEXTURE_2D, texid);
        glTexImage2D(GL_TEXTURE_2D, 0, ((g_Bpp==1) ? GL_LUMINANCE : GL_BGRA), 
                    imWidth, imHeight,  0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
    }
}

void loadDefaultImage( int argc, char ** argv ) {

    printf("Loading input image: lena.pgm\n");
    const char* image_filename = "lena.pgm";
    char* image_path = cutFindFilePath(image_filename, argv[0] );
    if (image_path == NULL) {
       printf( "FunctionPointers unable to find and load image file <%s>.\nExiting...\n", image_filename);
       shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
    }
    initializeData( image_path, argc, argv );
    cutFree( image_path );
}

void printHelp()
{
    printf("\nUsage: FunctionPointers <options>\n");
    printf("\t\t-fbo    (render using FBO display path)\n");
    printf("\t\t-qatest (automatic validation)\n");
    printf("\t\t-file = filename.pgm (image files for input)\n\n");
    bQuit = true;
}

void initGL(int *argc, char** argv)
{
    glutInit( argc, argv );    
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("Function Pointers [CUDA Edge Detection]n");

    glewInit();

    if (g_bFBODisplay) {
        if (!glewIsSupported( "GL_VERSION_2_0 GL_ARB_fragment_program GL_EXT_framebuffer_object" )) {
            fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
            fprintf(stderr, "This sample requires:\n");
            fprintf(stderr, "  OpenGL version 2.0\n");
            fprintf(stderr, "  GL_ARB_fragment_program\n");
            fprintf(stderr, "  GL_EXT_framebuffer_object\n");
            cutilDeviceReset();
            shrQAFinishExit(*argc, (const char **)argv, QA_WAIVED);
        } 
    } else {
        if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
            fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
            fprintf(stderr, "This sample requires:\n");
            fprintf(stderr, "  OpenGL version 1.5\n");
            fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
            fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
            cutilDeviceReset();
            shrQAFinishExit(*argc, (const char **)argv, QA_WAIVED);
        }
    }
}

bool checkCUDAProfile(int dev, int min_runtime, int min_compute)
{
    int runtimeVersion = 0;     

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    fprintf(stderr,"\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    cudaRuntimeGetVersion(&runtimeVersion);
    fprintf(stderr,"  CUDA Runtime Version     :\t%d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
    fprintf(stderr,"  CUDA Compute Capability  :\t%d.%d\n", deviceProp.major, deviceProp.minor);

    if( runtimeVersion >= min_runtime && ((deviceProp.major<<4) + deviceProp.minor) >= min_compute ) {
        return true;
    } else {
        return false;
    }
}

int findCapableDevice(int argc, char **argv)
{
    int dev;
    int bestDev = -1;

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        printf( "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
        shrQAFinishExit(*pArgc, (const char **)pArgv, QA_FAILED);
    }

    if (deviceCount == 0)
        fprintf(stderr,"There is no device supporting CUDA.\n");
    else
        fprintf(stderr,"Found %d CUDA Capable Device(s).\n", deviceCount);

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if( checkCUDAProfile( dev, MIN_RUNTIME_VERSION, MIN_COMPUTE_VERSION ) ) {
            fprintf(stderr,"\nFound CUDA Capable Device %d: \"%s\"\n", dev, deviceProp.name );
            if( bestDev == -1 ) { 
                bestDev = dev;
                fprintf(stderr, "Setting active device to %d\n", bestDev );
            }
        }
    }

    if( bestDev == -1 ) {
        fprintf(stderr, "\nNo configuration with available capabilities was found.  Test has been waived.\n");
        fprintf(stderr, "This SDK sample has minimum requirements:\n");
		fprintf(stderr, "\tCUDA Compute Capability >= %d.%d is required\n", MIN_COMPUTE_VERSION/16, MIN_COMPUTE_VERSION%16);
        fprintf(stderr, "\tCUDA Runtime Version    >= %d.%d is required\n", MIN_RUNTIME_VERSION/1000, (MIN_RUNTIME_VERSION%100)/10);
//        fprintf(stderr, "PASSED\n");
    }
    return bestDev;
}

void checkDeviceMeetComputeSpec( int argc, char **argv )
{
	int device = 0;
	cudaGetDevice( &device );
	if( checkCUDAProfile( device, MIN_RUNTIME_VERSION, MIN_COMPUTE_VERSION) ) {
	   fprintf(stderr,"\nCUDA Capable Device %d, meets minimum required specs.\n", device );
	} else {
       fprintf(stderr, "\nNo configuration with minimum compute capabilities found.  Exiting...\n");
       fprintf(stderr, "This sample requires:\n");
       fprintf(stderr, "\tCUDA Compute Capability >= %d.%d is required\n", MIN_COMPUTE_VERSION/16, MIN_COMPUTE_VERSION%16);
       fprintf(stderr, "\tCUDA Runtime Version    >= %d.%d is required\n", MIN_RUNTIME_VERSION/1000, (MIN_RUNTIME_VERSION%100)/10);
       cutilDeviceReset();
       shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
	}
}


void runAutoTest(int argc, char **argv)
{
    printf("[%s] (automated testing w/ readback)\n", sSDKsample);

    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) 
    {
       int device = cutilDeviceInit(argc, argv);
       if (device < 0) {
            printf("No CUDA Capable devices found, exiting...\n");
            shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
       }
	   checkDeviceMeetComputeSpec( argc, argv );
    } else {
       int dev = findCapableDevice(argc, argv);
       if( dev != -1 ) 
          cudaSetDevice( dev );
       else {
          cutilDeviceReset();
		  shrQAFinishExit2(g_bQAReadback, *pArgc, (const char **)pArgv, QA_PASSED);
       }
    }

    loadDefaultImage( argc, argv );

    if (argc > 1) {
        char *filename;
        if (cutGetCmdLineArgumentstr(argc, (const char **)argv, "file", &filename)) {
            initializeData(filename, argc, argv);
        }
    } else {
        loadDefaultImage( argc, argv );
    }

    g_CheckRender       = new CheckBackBuffer(imWidth, imHeight, sizeof(Pixel), false);
    g_CheckRender->setExecPath(argv[0]);

    Pixel *d_result;
    cutilSafeCall( cudaMalloc( (void **)&d_result, imWidth*imHeight*sizeof(Pixel)) );

    while (g_SobelDisplayMode <= 2) 
    {
        printf("AutoTest: %s <%s>\n", sSDKsample, filterMode[g_SobelDisplayMode]);

        sobelFilter(d_result, imWidth, imHeight, g_SobelDisplayMode, imageScale, blockOp, pointOp );

        cutilSafeCall( cutilDeviceSynchronize() );

        cudaMemcpy(g_CheckRender->imageData(), d_result, imWidth*imHeight*sizeof(Pixel), cudaMemcpyDeviceToHost);

        g_CheckRender->savePGM(sOriginal[g_Index], false, NULL);

        if (!g_CheckRender->PGMvsPGM(sOriginal[g_Index], sReference[g_Index], MAX_EPSILON_ERROR, 0.15f)) {
            g_TotalErrors++;
        }
        g_Index++;
        g_SobelDisplayMode = (SobelDisplayMode)g_Index;
    }

    cutilSafeCall( cudaFree( d_result ) );
    delete g_CheckRender;

    shrQAFinishExit(argc, (const char **)argv, (!g_TotalErrors ? QA_PASSED : QA_FAILED) );
}


int main(int argc, char** argv) 
{
	pArgc = &argc;
	pArgv = argv;

	shrQAStart(argc, argv);

    if (argc > 1) {
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "help")) {
            printHelp();
        }
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "qatest") ||
            cutCheckCmdLineFlag(argc, (const char **)argv, "noprompt"))
        {
            g_bQAReadback = true;
            fpsLimit = frameCheckNumber;
        }
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "glverify"))
		{
            g_bOpenGLQA = true;
            fpsLimit = frameCheckNumber;
        }
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "fbo")) {
            g_bFBODisplay = true;
            fpsLimit = frameCheckNumber;
        }
    }
	

    if (g_bQAReadback) 
    {
        runAutoTest(argc, argv);
    } 
    else 
    {
        if ( cutCheckCmdLineFlag(argc, (const char **)argv, "device")) {
             printf("   This SDK does not explicitly support -device=n when running with OpenGL.\n");
             printf("   When specifying -device=n (n=0,1,2,....) the sample must not use OpenGL.\n");
             printf("   See details below to run without OpenGL:\n\n");
             printf(" > %s -device=n -qatest\n\n", argv[0]);
             printf("exiting...\n");
             shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
        }

        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        initGL( &argc, argv );

        //cudaGLSetGLDevice (cutGetMaxGflopsDeviceId() );
        int dev = findCapableDevice(argc, argv);
        if( dev != -1 ) {
            cudaGLSetGLDevice( dev );
        } else {
            shrQAFinishExit2(g_bQAReadback, *pArgc, (const char **)pArgv, QA_PASSED);
        }

        cutilCheckError(cutCreateTimer(&timer));
        cutilCheckError(cutResetTimer(timer));  
     
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutReshapeFunc(reshape);

        if (g_bOpenGLQA) {
            loadDefaultImage( argc, argv );
        }

        if (argc > 1) {
            char *filename;
            if (cutGetCmdLineArgumentstr(argc, (const char **)argv, "file", &filename)) {
                initializeData(filename, argc, argv);
            }
        } else {
            loadDefaultImage( argc, argv );
        }


        // If code is not printing the USage, then we execute this path.
        if (!bQuit) {
            if (g_bOpenGLQA) {
                g_CheckRender = new CheckBackBuffer(wWidth, wHeight, 4);
                g_CheckRender->setPixelFormat(GL_BGRA);
                g_CheckRender->setExecPath(argv[0]);
                g_CheckRender->EnableQAReadback(true);
            }

            printf("I: display Image (no filtering)\n");
            printf("T: display Sobel Edge Detection (Using Texture)\n");
            printf("S: display Sobel Edge Detection (Using SMEM+Texture)\n");
            printf("Use the '-' and '=' keys to change the brightness.\n");
			printf("b: switch block filter operation (mean/Sobel)\n");
			printf("p: switch point filter operation (threshold on/off)\n");
            fflush(stdout);
            atexit(cleanup); 
            glutTimerFunc(REFRESH_DELAY, timerEvent,0);
            glutMainLoop();
        }
    }

    cutilDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
}
