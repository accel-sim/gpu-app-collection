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
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <rendercheck_gl.h>
#include <sdkHelper.h>    // includes cuda.h and cuda_runtime_api.h
#include <shrQATest.h>    // standard utility and system includes
#include <vector_types.h>


#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD		  0.30f
#define REFRESH_DELAY	  10 //ms

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "simpleGL.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_simpleGL.ppm",
    NULL
};


////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_Verify = false;
bool g_bQAReadback = false;
bool g_bGLVerify   = false;

int *pArgc = NULL;
char **pArgv = NULL;

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// kernels
//#include <simpleGL_kernel.cu>

extern "C" 
void launch_kernel(float4* pos, unsigned int mesh_width, unsigned int mesh_height, float time);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char** argv);
void cleanup();

// GL functionality
bool initGL(int *argc, char** argv);
void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, 
	       unsigned int vbo_res_flags);
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void runAutoTest();
void checkResultCuda(int argc, char** argv, const GLuint& vbo);

const char *SDK_name = "simpleGL (VBO)";

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

bool checkHW(char *name, char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType))) {
       return true;
    } else {
       return false;
    }
}

int findGraphicsGPU(char *name)
{
    int nGraphicsGPU = 0;
    int deviceCount = 0;
    bool bFoundGraphics = false;
    char firstGraphicsName[256], temp[256];

    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        printf( "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
        shrQAFinishExit(*pArgc, (const char **)pArgv, QA_FAILED);
    }
    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("> There are no device(s) supporting CUDA\n");
		return false;
    } else {
		printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }
    for (int dev = 0; dev < deviceCount; ++dev) {
		bool bGraphics = !checkHW(temp, "Tesla", dev);
		printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);
		if (bGraphics) {
			if (!bFoundGraphics) {
				strcpy(firstGraphicsName, temp);
			}
			nGraphicsGPU++;
		}
	}
	if (nGraphicsGPU) {
		strcpy(name, firstGraphicsName);
	} else {
		strcpy(name, "this hardware");
	}
	return nGraphicsGPU;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	pArgc = &argc;
	pArgv = argv;

	shrQAStart(argc, argv);

    if (argc > 1) {
        if (checkCmdLineFlag(argc, (const char **)argv, "qatest") ||
            checkCmdLineFlag(argc, (const char **)argv, "noprompt")) 
        {
            printf("- (automated test no-OpenGL)\n");
            g_bQAReadback = true;
            //			g_bGLVerify = true;	
            fpsLimit = frameCheckNumber;
        } else if (checkCmdLineFlag(argc, (const char **)argv, "glverify")) {
            printf("- (automated test OpenGL rendering)\n");
            g_bGLVerify = true;	
            fpsLimit = frameCheckNumber;
        }
    }
    printf("\n");
    
    runTest(argc, argv);
    
    cudaDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, (g_TotalErrors == 0) ? QA_PASSED : QA_FAILED);
}

void AutoQATest()
{
    if (g_CheckRender && g_CheckRender->IsQAReadback()) {
        char temp[256];
        sprintf(temp, "AutoTest: Cuda GL Interop (VBO)");
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
        sprintf(fps, "%sCuda GL Interop (VBO): %3.1f fps (Max 100Hz)", 
                ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        if (g_CheckRender && !g_CheckRender->IsQAReadback()) 
	    fpsLimit = (int)MAX(ifps, 1.f);

        sdkResetTimer(&timer);  

        AutoQATest();
    }
 }

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);

	// initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported("GL_VERSION_2_0 ")) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char** argv)
{
    // Create the CUTIL timer
    sdkCreateTimer( &timer );
    
    // command line mode only
    if (g_bQAReadback) {
        // This will pick the best possible CUDA capable device
        int devID = findCudaDevice((const int)argc, (const char **)argv);
		
		// create VBO
		createVBO(NULL, NULL, 0);
    } else {
		// First initialize OpenGL context, so we can properly set the GL for CUDA.
		// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
		if (false == initGL(&argc, argv)) {
			return false;
		}
		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		if( checkCmdLineFlag(argc, (const char**)argv, "device") ) {
			gpuGLDeviceInit(argc, argv);
		} else {
			cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );
		}
		
		// register callbacks
		glutDisplayFunc(display);
		glutKeyboardFunc(keyboard);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
		
		// create VBO
		createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    }
    
    if (g_bQAReadback) {
        g_CheckRender = new CheckBackBuffer(window_width, window_height, 4, false);
        g_CheckRender->setPixelFormat(GL_RGBA);
        g_CheckRender->setExecPath(argv[0]);
        g_CheckRender->EnableQAReadback(true);

        runAutoTest();
    } else {
        if (g_bGLVerify) {
            g_CheckRender = new CheckBackBuffer(window_width, window_height, 4);
            g_CheckRender->setPixelFormat(GL_RGBA);
            g_CheckRender->setExecPath(argv[0]);
            g_CheckRender->EnableQAReadback(true);
        }

        // run the cuda part
        runCuda(&cuda_vbo_resource);
    }

    // check result of Cuda step
    checkResultCuda(argc, argv, vbo);

    if (!g_bQAReadback) {
	    atexit(cleanup);
    	
	    // start rendering mainloop
	    glutMainLoop();
    }
	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&dptr, vbo));
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes; 
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,  
						       *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    // execute the kernel
    //    dim3 block(8, 8, 1);
    //    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    //    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

    launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

    // unmap buffer object
    // DEPRECATED: checkCudaErrors(cudaGLUnmapBufferObject(vbo));
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest()
{
    // execute the kernel
    launch_kernel((float4 *)d_vbo_buffer, mesh_width, mesh_height, g_fAnim);

    cudaDeviceSynchronize();
    getLastCudaError("launch_kernel failed");

    checkCudaErrors( cudaMemcpy( g_CheckRender->imageData(), d_vbo_buffer, mesh_width*mesh_height*sizeof(float), cudaMemcpyDeviceToHost) );
    g_CheckRender->dumpBin((void *)g_CheckRender->imageData(), mesh_width*mesh_height*sizeof(float), "simpleGL.bin");
    if (!g_CheckRender->compareBin2BinFloat("simpleGL.bin", "ref_simpleGL.bin", mesh_width*mesh_height*sizeof(float), MAX_EPSILON_ERROR, THRESHOLD))
       g_TotalErrors++;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, 
	       unsigned int vbo_res_flags)
{
    if (vbo) {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // register this buffer object with CUDA
        // DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(*vbo));
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

        SDK_CHECK_ERROR_GL();
    } else {
        checkCudaErrors( cudaMalloc( (void **)&d_vbo_buffer, mesh_width*mesh_height*4*sizeof(float) ) );
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res)
{
    if (vbo) {
	// unregister this buffer object with CUDA
	//DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(*pbo));
	cudaGraphicsUnregisterResource(vbo_res);
	
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	
	*vbo = 0;
    } else {
	cudaFree(d_vbo_buffer);
	d_vbo_buffer = NULL;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    if (g_CheckRender && g_CheckRender->IsQAReadback() && g_Verify) {
        // readback for QA testing
        printf("> (Frame %d) Readback BackBuffer\n", frameCount);
        g_CheckRender->readback( window_width, window_height );
        g_CheckRender->savePPM(sOriginal[g_Index], true, NULL);
        if (!g_CheckRender->PPMvsPPM(sOriginal[g_Index], sReference[g_Index], MAX_EPSILON_ERROR, 0.15f)) {
            g_TotalErrors++;
        }
        g_Verify = false;
    }

    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

void cleanup()
{
    sdkDeleteTimer( &timer );

    deleteVBO(&vbo, cuda_vbo_resource);

    if (g_CheckRender) {
        delete g_CheckRender; g_CheckRender = NULL;
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case(27) :
        exit(0);
        break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char** argv, const GLuint& vbo)
{
    if (!d_vbo_buffer) {
	    //DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(vbo));
	    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    	
	    // map buffer object
	    glBindBuffer(GL_ARRAY_BUFFER_ARB, vbo );
	    float* data = (float*) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    	
	    // check result
	    if(checkCmdLineFlag(argc, (const char**) argv, "regression")) {
	        // write file for regression test
            sdkWriteFile<float>("./data/regression.dat",
                                data, mesh_width * mesh_height * 3, 0.0, false);
	    }
    	
	    // unmap GL buffer object
	    if(! glUnmapBuffer(GL_ARRAY_BUFFER)) {
	        fprintf(stderr, "Unmap buffer failed.\n");
	        fflush(stderr);
	    }
    	
	    //DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(vbo));
	    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, 
                        cudaGraphicsMapFlagsWriteDiscard));
    	
	    SDK_CHECK_ERROR_GL();
    }
}
