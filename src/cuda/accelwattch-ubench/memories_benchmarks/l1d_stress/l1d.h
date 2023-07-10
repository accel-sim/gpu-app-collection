/*
 * l1d.h
 *
 *  Created on: 2012-06-20
 *      Author: tayler
 */

#ifndef L1D_H_
#define L1D_H_

/*
#define asm_ld(dst_reg, addr, tid) int *new_addr = addr+tid; int d;\
			asm volatile ("ld.global.u32 %0, [%1];" : "=r"(d) : "l"(new_addr) : "memory"); \
			dst_reg +=d; \
			asm volatile ("ld.global.u32 %0, [%1];" : "=r"(d) : "l"(new_addr) : "memory"); \
			dst_reg +=d; \
			asm volatile ("ld.global.u32 %0, [%1];" : "=r"(d) : "l"(new_addr) : "memory"); \
			dst_reg +=d; \
			asm volatile ("ld.global.u32 %0, [%1];" : "=r"(d) : "l"(new_addr) : "memory"); \
			dst_reg +=d; \
			asm volatile ("ld.global.u32 %0, [%1];" : "=r"(d) : "l"(new_addr) : "memory"); \
*/

#define asm_ld(dst_reg, addr, tid) int *new_addr = addr+tid;	\
			asm volatile ( "ld.global.u32 %0, [%1];"			\
				"ld.global.u32 %0, [%1];"						\
				"ld.global.u32 %0, [%1];"						\
				"ld.global.u32 %0, [%1];"						\
				"ld.global.u32 %0, [%1];"						\
				"ld.global.u32 %0, [%1];"						\
				"ld.global.u32 %0, [%1];"						\
				"ld.global.u32 %0, [%1];"						\
				: "=r"(dst_reg) : "l"(new_addr) : "memory");	\



//#define ld(dst_reg, addr, tid)


#define add10(result, add) 	result += add; \
							result += add; \
							result += add; \
							result += add; \
							result += add; \
							result += add; \
							result += add; \
							result += add; \
							result += add; \
							result += add; \


#define ld5_add5(sum, addr, start) 	\
{										\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 5;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 6;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 7;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 8;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 9;							\
}										\


#define ld10(sum, addr, start) 	\
{								\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
} 								\

#define ld50_add50(sum, addr, start) 	\
{										\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 5;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 6;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 7;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 8;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 9;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 5;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 6;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 7;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 8;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 9;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 5;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 6;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 7;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 8;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 9;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 5;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 6;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 7;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 8;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 9;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 5;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 6;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 7;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 8;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 9;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 5;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 6;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 7;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 8;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 9;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 5;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 6;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 7;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 8;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 9;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 5;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 6;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 7;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 8;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 9;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 5;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 6;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 7;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 8;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 9;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 5;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 6;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 7;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 8;							\
	sum += addr[start];					\
	start+=THREADS_PER_BLOCK;			\
	sum += 9;							\
}										\

#define ld100(sum, addr, start) 	\
{								\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
	start+=THREADS_PER_BLOCK;	\
	sum += addr[start];			\
} 								\



#endif /* L1D_H_ */
