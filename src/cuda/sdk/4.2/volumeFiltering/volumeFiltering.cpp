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
CUDA 3D Volume Filtering sample

This sample loads a 3D volume from disk and displays it using
ray marching and 3D textures.

Note - this is intended to be an example of using 3D textures
in CUDA, not an optimized volume renderer.

Changes
sgg 22/3/2010
- updated to use texture for display instead of glDrawPixels.
- changed to render from front-to-back rather than back-to-front.
*/

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_gl_inline.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h
#include <rendercheck_gl.h>

// CUDA Includes
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// Shared Library Test Functions
#include <shrUtils.h>
#include <shrQATest.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#define MIN_RUNTIME_VERSION 4010
#define MIN_COMPUTE_VERSION 0x20

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
  "volumefilter.ppm",
  NULL
};

const char *sReference[] =
{
  "ref_volumefilter.ppm",
  NULL
};

const char *sSDKsample = "CUDA 3D Volume Filtering";


#include "volume.h"
#include "volumeFilter.h"
#include "volumeRender.h"

char *volumeFilename = "Bucky.raw";
cudaExtent volumeSize = make_cudaExtent(32, 32, 32);

uint width = 512, height = 512;
dim3 blockSize(16, 16);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool  linearFiltering = true;
bool  preIntegrated = true;
unsigned int animationTimer = 0;

float   filterFactor = 0.0f;
bool    filterAnimation = true;
int     filterIterations = 2;
float   filterTimeScale = 0.001f;
float   filterBias = 0.0f;
float4  filterWeights[3*3*3];

Volume  volumeOriginal;
Volume  volumeFilter0;
Volume  volumeFilter1;

GLuint pbo = 0;           // OpenGL pixel buffer object
GLuint volumeTex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

unsigned int timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_Verify = false;
bool g_bQAReadback = false;
bool g_bQAGLVerify = false;
bool g_bFBODisplay = false;

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

int *pArgc;
char **pArgv;

#define MAX(a,b) ((a > b) ? a : b)

//////////////////////////////////////////////////////////////////////////
// QA RELATED

void AutoQATest()
{
  if (g_CheckRender && g_CheckRender->IsQAReadback()) {
    char temp[256];
    sprintf(temp, "AutoTest: CUDA 3D Volume Filtering");
    glutSetWindowTitle(temp);
    shrQAFinishExit2(false, *pArgc, (const char **)pArgv, QA_PASSED);
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
    sprintf(fps, "%sCUDA 3D Volume Filtering: %3.1f fps", 
      ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), ifps);  

    glutSetWindowTitle(fps);
    fpsCount = 0; 
    if (g_CheckRender && !g_CheckRender->IsQAReadback()) fpsLimit = (int)MAX(ifps, 1.f);

    cutilCheckError(cutResetTimer(timer));  

    AutoQATest();
  }
}

//////////////////////////////////////////////////////////////////////////
// 3D FILTER

static float filteroffsets[3*3*3][3] = {
  {-1,-1,-1},{ 0,-1,-1},{ 1,-1,-1},
  {-1, 0,-1},{ 0, 0,-1},{ 1, 0,-1},
  {-1, 1,-1},{ 0, 1,-1},{ 1, 1,-1},

  {-1,-1, 0},{ 0,-1, 0},{ 1,-1, 0},
  {-1, 0, 0},{ 0, 0, 0},{ 1, 0, 0},
  {-1, 1, 0},{ 0, 1, 0},{ 1, 1, 0},

  {-1,-1, 1},{ 0,-1, 1},{ 1,-1, 1},
  {-1, 0, 1},{ 0, 0, 1},{ 1, 0, 1},
  {-1, 1, 1},{ 0, 1, 1},{ 1, 1, 1},
};

static float filterblur[3*3*3] = {
  0,1,0,
  1,2,1,
  0,1,0,

  1,2,1,
  2,4,2,
  1,2,1,

  0,1,0,
  1,2,1,
  0,1,0,
};
static float filtersharpen[3*3*3] = {
  0,0,0,
  0,-2,0,
  0,0,0,

  0,-2,0,
  -2,15,-2,
  0,-2,0,

  0,0,0,
  0,-2,0,
  0,0,0,
};

static float filterpassthru[3*3*3] = {
  0,0,0,
  0,0,0,
  0,0,0,

  0,0,0,
  0,1,0,
  0,0,0,

  0,0,0,
  0,0,0,
  0,0,0,
};

void FilterKernel_init(){
  float sumblur = 0.0f;
  float sumsharpen = 0.0f;
  for (int i = 0; i < 3*3*3; i++){
    sumblur += filterblur[i];
    sumsharpen += filtersharpen[i];
  }
  
  for (int i = 0; i < 3*3*3; i++){
    filterblur[i] /= sumblur;
    filtersharpen[i] /= sumsharpen;

    filterWeights[i].x = filteroffsets[i][0];
    filterWeights[i].y = filteroffsets[i][1];
    filterWeights[i].z = filteroffsets[i][2];
  }
}

void FilterKernel_update(float blurfactor)
{
  if (blurfactor > 0.0f){
    for (int i = 0; i < 3*3*3; i++){
      filterWeights[i].w = filterblur[i] * blurfactor + filterpassthru[i] * (1.0f - blurfactor);
    }
  }
  else{
    blurfactor = -blurfactor;
    for (int i = 0; i < 3*3*3; i++){
      filterWeights[i].w = filtersharpen[i] * blurfactor + filterpassthru[i] * (1.0f - blurfactor);
    }
  }
  
}

void filter()
{
  if (filterAnimation){
    filterFactor = cosf(cutGetTimerValue(animationTimer) * filterTimeScale);
  }

  FilterKernel_update(filterFactor);

  Volume* volumeRender = VolumeFilter_runFilter(&volumeOriginal,&volumeFilter0,&volumeFilter1,
    filterIterations, 3*3*3,filterWeights,filterBias);

  VolumeRender_setVolume(volumeRender);

}

//////////////////////////////////////////////////////////////////////////
// RENDERING

// render image using CUDA
void render()
{
  
  VolumeRender_copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

  // map PBO to get CUDA device pointer
  uint *d_output;
  // map PBO to get CUDA device pointer
  cutilSafeCall(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
  size_t num_bytes; 
  cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  
    cuda_pbo_resource));
  //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

  // clear image
  cutilSafeCall(cudaMemset(d_output, 0, width*height*4));

  // call CUDA kernel, writing results to PBO
  VolumeRender_render(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale);

  cutilCheckMsg("render kernel failed");

  cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
  cutilCheckError(cutStartTimer(timer));  

  // use OpenGL to build view matrix
  GLfloat modelView[16];
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
  glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
  glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
  glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
  glPopMatrix();

  invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
  invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
  invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

  filter();
  render();

  // display results
  glClear(GL_COLOR_BUFFER_BIT);

  // draw image from PBO
  glDisable(GL_DEPTH_TEST);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  // draw using texture
  // copy from pbo to texture
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
  glBindTexture(GL_TEXTURE_2D, volumeTex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  // draw textured quad
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glTexCoord2f(0, 0); glVertex2f(0, 0);
  glTexCoord2f(1, 0); glVertex2f(1, 0);
  glTexCoord2f(1, 1); glVertex2f(1, 1);
  glTexCoord2f(0, 1); glVertex2f(0, 1);
  glEnd();

  glDisable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, 0);


  if (g_CheckRender && g_CheckRender->IsQAReadback() && g_Verify) {
    // readback for QA testing
    shrLog("\n> (Frame %d) Readback BackBuffer\n", frameCount);
    g_CheckRender->readback( width, height );
    g_CheckRender->savePPM(sOriginal[g_Index], true, NULL);
    if (!g_CheckRender->PPMvsPPM(sOriginal[g_Index], sReference[g_Index], MAX_EPSILON_ERROR, THRESHOLD)) {
      g_TotalErrors++;
    }
    g_Verify = false;
  }
  glutSwapBuffers();
  glutReportErrors();

  cutilCheckError(cutStopTimer(timer));  

  computeFPS();
}

void idle()
{
  glutPostRedisplay();
}

//////////////////////////////////////////////////////////////////////////
// LOGIC

void keyboard(unsigned char key, int x, int y)
{
  switch(key) {
case 27:
  shrQAFinishExit2(false, *pArgc, (const char **)pArgv, QA_PASSED);
  break;
case ' ':
  filterAnimation = !filterAnimation;
  if (!filterAnimation){
    cutStopTimer(animationTimer);
  }
  else{
    cutStartTimer(animationTimer);
  }
  break;
case 'f':
  linearFiltering = !linearFiltering;
  VolumeRender_setTextureFilterMode(linearFiltering);
  break;
case 'p':
  preIntegrated = !preIntegrated;
  VolumeRender_setPreIntegrated(preIntegrated);
  break;
case '+':
  density += 0.01f;
  break;
case '-':
  density -= 0.01f;
  break;

case ']':
  brightness += 0.1f;
  break;
case '[':
  brightness -= 0.1f;
  break;

case ';':
  transferOffset += 0.01f;
  break;
case '\'':
  transferOffset -= 0.01f;
  break;

case '.':
  transferScale += 0.01f;
  break;
case ',':
  transferScale -= 0.01f;
  break;

default:
  break;
  }
  shrLog("density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f\n", density, brightness, transferOffset, transferScale);
  glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
  if (state == GLUT_DOWN)
    buttonState  |= 1<<button;
  else if (state == GLUT_UP)
    buttonState = 0;

  ox = x; oy = y;
  glutPostRedisplay();
}

void motion(int x, int y)
{
  float dx, dy;
  dx = (float)(x - ox);
  dy = (float)(y - oy);

  if (buttonState == 4) {
    // right = zoom
    viewTranslation.z += dy / 100.0f;
  } 
  else if (buttonState == 2) {
    // middle = translate
    viewTranslation.x += dx / 100.0f;
    viewTranslation.y -= dy / 100.0f;
  }
  else if (buttonState == 1) {
    // left = rotate
    viewRotation.x += dy / 5.0f;
    viewRotation.y += dx / 5.0f;
  }

  ox = x; oy = y;
  glutPostRedisplay();
}

//////////////////////////////////////////////////////////////////////////
// SAMPLE INIT/DEINIT

static int iDivUp(int a, int b){
  return (a % b != 0) ? (a / b + 1) : (a / b);
}
void initPixelBuffer();
void reshape(int w, int h)
{
  width = w; height = h;
  initPixelBuffer();

  // calculate new grid size
  gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

  glViewport(0, 0, w, h);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}


void initGL(int *argc, char **argv)
{
  // initialize GLUT callback functions
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowSize(width, height);
  glutCreateWindow("CUDA 3D Volume Filtering");

  glewInit();
  if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
    shrLog("Required OpenGL extensions missing.");
    shrQAFinishExit(*argc, (const char **)argv, QA_WAIVED);
  }
}

void initPixelBuffer()
{
  if (pbo) {
    // unregister this buffer object from CUDA C
    cutilSafeCall(cudaGraphicsUnregisterResource(cuda_pbo_resource));

    // delete old buffer
    glDeleteBuffersARB(1, &pbo);
    glDeleteTextures(1, &volumeTex);
  }

  // create pixel buffer object for display
  glGenBuffersARB(1, &pbo);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
  glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  // register this buffer object with CUDA
  cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));    

  // create texture for display
  glGenTextures(1, &volumeTex);
  glBindTexture(GL_TEXTURE_2D, volumeTex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glBindTexture(GL_TEXTURE_2D, 0);
}

//////////////////////////////////////////////////////////////////////////

void cleanup()
{
  cutilCheckError( cutDeleteTimer( timer));
  cutilCheckError( cutDeleteTimer( animationTimer ) );

  Volume_deinit(&volumeOriginal);
  Volume_deinit(&volumeFilter0);
  Volume_deinit(&volumeFilter1);
  VolumeRender_deinit();

  if (pbo) {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffersARB(1, &pbo);
    glDeleteTextures(1, &volumeTex);
  }

  if (g_CheckRender) {
    delete g_CheckRender;
    g_CheckRender = NULL;
  }
}

// Load raw data from disk
void *loadRawFile(char *filename, size_t size)
{
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Error opening file '%s'\n", filename);
    return 0;
  }

  void *data = malloc(size);
  size_t read = fread(data, 1, size, fp);
  fclose(fp);

  shrLog("Read '%s', %d bytes\n", filename, read);

  return data;
}

void initData(int argc, char**argv)
{
  // parse arguments
  char *filename;
  if (cutGetCmdLineArgumentstr( argc, (const char**) argv, "file", &filename)) {
    volumeFilename = filename;
  }
  int n;
  if (cutGetCmdLineArgumenti( argc, (const char**) argv, "size", &n)) {
    volumeSize.width = volumeSize.height = volumeSize.depth = n;
  }
  if (cutGetCmdLineArgumenti( argc, (const char**) argv, "xsize", &n)) {
    volumeSize.width = n;
  }
  if (cutGetCmdLineArgumenti( argc, (const char**) argv, "ysize", &n)) {
    volumeSize.height = n;
  }
  if (cutGetCmdLineArgumenti( argc, (const char**) argv, "zsize", &n)) {
    volumeSize.depth = n;
  }

  char* path = shrFindFilePath(volumeFilename, argv[0]);
  if (path == 0) {
    shrLog("Error finding file '%s'\n", volumeFilename);
    shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
  }

  size_t size = volumeSize.width*volumeSize.height*volumeSize.depth*sizeof(VolumeType);
  void *h_volume = loadRawFile(path, size);

  FilterKernel_init();
  Volume_init(&volumeOriginal,volumeSize, h_volume, 0);
  free(h_volume);
  Volume_init(&volumeFilter0, volumeSize, NULL, 1);
  Volume_init(&volumeFilter1, volumeSize, NULL, 1);
  VolumeRender_init();
  VolumeRender_setPreIntegrated(preIntegrated);
  VolumeRender_setVolume(&volumeOriginal);

  cutilCheckError( cutCreateTimer( &timer ) );
  cutilCheckError( cutCreateTimer( &animationTimer ) );
  cutilCheckError( cutStartTimer(animationTimer) );

  // calculate new grid size
  gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}

//////////////////////////////////////////////////////////////////////////
// CUDA DEVICE

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

//////////////////////////////////////////////////////////////////////////
// AUTOMATIC TESTING

void runAutoTest(int argc, char** argv)
{
  g_CheckRender = new CheckBackBuffer(width, height, 4, false);
  g_CheckRender->setPixelFormat(GL_RGBA);
  g_CheckRender->setExecPath(argv[0]);
  g_CheckRender->EnableQAReadback(true);

  uint *d_output;
  cutilSafeCall(cudaMalloc((void**)&d_output, width*height*sizeof(uint)));
  cutilSafeCall(cudaMemset(d_output, 0, width*height*sizeof(uint)));

  float modelView[16] = 
  {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 4.0f, 1.0f
  };

  invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
  invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
  invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

  // call CUDA kernel, writing results to PBO
  VolumeRender_copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);
  filterAnimation = false;

  // Start timer 0 and process n loops on the GPU 
  int nIter = 10;
  float scale = 2.0f/float(nIter-1);
  for (int i = -1; i < nIter; i++)
  {
    if( i == 0 ) {
      cutilDeviceSynchronize();
      cutStartTimer(timer); 
    }

    filterFactor = (float(i) * scale) - 1.0f;
    filterFactor = -filterFactor;
    filter();
    VolumeRender_render(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale);
  }
  cutilDeviceSynchronize();
  cutStopTimer(timer);
  // Get elapsed time and throughput, then log to sample and master logs
  double dAvgTime = cutGetTimerValue(timer)/(nIter * 1000.0);
  shrLogEx(LOGBOTH | MASTER, 0, "volumeFiltering, Throughput = %.4f MTexels/s, Time = %.5f s, Size = %u Texels, NumDevsUsed = %u, Workgroup = %u\n", 
    (1.0e-6 * width * height)/dAvgTime, dAvgTime, (width * height), 1, blockSize.x * blockSize.y); 


  cutilCheckMsg("Error: kernel execution FAILED");
  cutilSafeCall( cutilDeviceSynchronize() );

  cutilSafeCall( cudaMemcpy(g_CheckRender->imageData(), d_output, width*height*4, cudaMemcpyDeviceToHost) );
  g_CheckRender->savePPM(sOriginal[g_Index], true, NULL);

  bool bTestResult = g_CheckRender->PPMvsPPM(sOriginal[g_Index], sReference[g_Index], MAX_EPSILON_ERROR, THRESHOLD);

  cutilSafeCall( cudaFree(d_output) );
  cleanup();

  shrQAFinishExit(argc, (const char **)argv, (bTestResult ? QA_PASSED : QA_FAILED) );
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void printHelp()
{
  printf("\nUsage: volumeFiltering <options>\n");
  printf("\t\t-qatest (automatic validation)\n");
  printf("\t\t-file = filename.raw (volume file for input)\n\n");
  printf("\t\t-size = 64 (volume size, isotropic)\n\n");
  printf("\t\t-xsize = 128 (volume size, anisotropic)\n\n");
  printf("\t\t-ysize = 128 (volume size, anisotropic)\n\n");
  printf("\t\t-zsize = 32 (volume size, anisotropic)\n\n");
}

int
main( int argc, char** argv) 
{
  pArgc = &argc;
  pArgv = argv;

  shrQAStart(argc, argv);

  //start logs

  if (cutCheckCmdLineFlag(argc, (const char **)argv, "help")) 
  {
    printHelp();
    shrQAFinishExit(argc, (const char **)argv, QA_PASSED );
  }

  if (cutCheckCmdLineFlag(argc, (const char **)argv, "qatest") ||
      cutCheckCmdLineFlag(argc, (const char **)argv, "noprompt")) 
  {
    g_bQAReadback = true;
    fpsLimit = frameCheckNumber;
  }

  if (cutCheckCmdLineFlag(argc, (const char **)argv, "glverify")) 
  {
    g_bQAGLVerify = true;
    fpsLimit = frameCheckNumber;
  }

  shrSetLogFileName ("volumeFiltering.txt");
  shrLog("%s Starting...\n\n", argv[0]); 

  if (g_bQAReadback) {
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
  } else {
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

    int dev = findCapableDevice(argc, argv);
    if( dev != -1 ) {
      cudaGLSetGLDevice( dev );
    } else {
      shrQAFinishExit2(g_bQAReadback, *pArgc, (const char **)pArgv, QA_PASSED);
    }
  }

  // load volume data
  initData(argc,argv); 

  shrLog(
    "Press \n"
    "  'SPACE'     to toggle animation\n"
    "  'p'         to toggle pre-integrated transfer function\n"
    "  '+' and '-' to change density (0.01 increments)\n"
    "  ']' and '[' to change brightness\n"
    "  ';' and ''' to modify transfer function offset\n"
    "  '.' and ',' to modify transfer function scale\n\n");

  if (g_bQAReadback) {
    runAutoTest(argc,argv);
  } else {
    // This is the normal rendering path for VolumeRender
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    initPixelBuffer();

    if (g_bQAGLVerify) {
      g_CheckRender = new CheckBackBuffer(width, height, 4);
      g_CheckRender->setPixelFormat(GL_RGBA);
      g_CheckRender->setExecPath(argv[0]);
      g_CheckRender->EnableQAReadback(true);
    }
    atexit(cleanup);

    glutMainLoop();
  }

  cutilDeviceReset();
  shrQAFinishExit(argc, (const char **)argv, QA_PASSED );
}
