#define REPEAT_L6(N)\
         REPEAT_L4(N);
//         REPEAT_L4(8000);\
//        REPEAT_L4(10000);\
//         REPEAT_L4(12000);
 
#define REPEAT_L5(N)\
          REPEAT_L4(N);\
          REPEAT_L4(2000);\
          REPEAT_L4(4000);\
          REPEAT_L4(6000);\
  
#define REPEAT_L4(N)\
         REPEAT_L3(N);\
         REPEAT_L3(N+400);\
         REPEAT_L3(N+800);\
         REPEAT_L3(N+1200);\
         REPEAT_L3(N+1600);
 

#define REPEAT_L3(N)\
	REPEAT_L2(N);\
	REPEAT_L2(N+80);\
	REPEAT_L2(N+160);\
	REPEAT_L2(N+240);\
	REPEAT_L2(N+320);


#define REPEAT_L2(N)\
	REPEAT_L1(N);\
	REPEAT_L1(N+16);\
	REPEAT_L1(N+32);\
	REPEAT_L1(N+48);\
	REPEAT_L1(N+64);
	

#define REPEAT_L1(N)\
C[((tid*LINE_SIZE)+((0+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((1+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((2+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((3+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((4+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((5+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((6+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((7+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((8+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((9+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((10+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((11+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((12+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((13+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((14+N)*LINE_SIZE))]=1;\
C[((tid*LINE_SIZE)+((15+N)*LINE_SIZE))]=1;

