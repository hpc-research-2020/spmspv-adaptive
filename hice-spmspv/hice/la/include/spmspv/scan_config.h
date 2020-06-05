#ifndef SCAN_CONFIG_H_
#define SCAN_CONFIG_H_

#define VEC 1 
#define LT 128
#define CT 8
#define REGP 2
#define LOGP 2
#define STEP_NUM (((REGP+LOGP)*CT)/VEC) 
#define REGISTER_SIZE (REGP*CT)
#define LOCALMEM_SIZE (LOGP*CT)

//#define NB_VEC_1
#define NB_REG_SIZE REGISTER_SIZE
#define NB_REG_GRP REGP
#define NB_LOCAL_GRP LOGP
#define NB_LOCAL_SIZE  (LOCALMEM_SIZE+1)

#define DYNAMIC_TASK

//for tail with fixed config.
#define TAIL_STEP_NUM 32
#define TAIL_NB_REG_GRP 1
#define TAIL_NB_REG_SIZE 16
#define TAIL_NB_LOCAL_GRP 1 
#define TAIL_NB_LOCAL_SIZE 17

#endif //SCAN_CONFIG_H_ 
