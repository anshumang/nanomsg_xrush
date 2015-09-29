/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @mainpage PIConGPU-Frame
 *
 * Project with HZDR for porting their PiC-code to a GPU cluster.
 *
 * \image html picongpu.jpg
 *
 * @author Heiko Burau, Rene Widera, Wolfgang Hoenig, Felix Schmitt, Axel Huebl, Michael Bussmann, Guido Juckeland
 */

//#include <thread>
#include <pthread.h>
#include "nn.h"
#include "pipeline.h"

pthread_mutex_t mutex1, mutex2;
pthread_cond_t condition1, condition2;
bool P_to_T = false, T_to_P = false;

extern pthread_mutex_t mutex1, mutex2;
extern pthread_cond_t condition1, condition2;
extern bool P_to_T, T_to_P;

extern int sockpull_t, sockpush_t, sockpull_p, sockpush_p;

//extern cudaIpcMemHandle_t *g_mem_handle;

// include the heap with the arguments given in the config
#include "mallocMC/mallocMC_utils.hpp"

// basic files for mallocMC
#include "mallocMC/mallocMC_overwrites.hpp"
#include "mallocMC/mallocMC_hostclass.hpp"

// Load all available policies for mallocMC
#include "mallocMC/CreationPolicies.hpp"
#include "mallocMC/DistributionPolicies.hpp"
#include "mallocMC/OOMPolicies.hpp"
#include "mallocMC/ReservePoolPolicies.hpp"
#include "mallocMC/AlignmentPolicies.hpp"

// configurate the CreationPolicy "Scatter"
struct ScatterConfig
{
    /* 2MiB page can hold around 256 particle frames */
    typedef boost::mpl::int_<2*1024*1024> pagesize;
    /* accessblocks, regionsize and wastefactor are not finale selected
       and might be performance sensitive*/
    typedef boost::mpl::int_<4> accessblocks;
    typedef boost::mpl::int_<8> regionsize;
    typedef boost::mpl::int_<2> wastefactor;
    /* resetfreedpages is used to minimize memory fracmentation while different
       frame sizes were used*/
    typedef boost::mpl::bool_<true> resetfreedpages;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
typedef mallocMC::Allocator<
mallocMC::CreationPolicies::Scatter<ScatterConfig>,
mallocMC::DistributionPolicies::Noop,
mallocMC::OOMPolicies::ReturnNull,
mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
mallocMC::AlignmentPolicies::Shrink<>
> ScatterAllocator;

//use ScatterAllocator to replace malloc/free
MALLOCMC_SET_ALLOCATOR_TYPE( ScatterAllocator );

#include <simulation_defines.hpp>
#include <mpi.h>
#include "ArgsParser.hpp"
#include "communication/manager_common.h"

#include <cupti.h>
#include <string>
#include <sstream>
#include <fstream>
#include <sys/mman.h>

using namespace PMacc;
using namespace picongpu;

/*! start of PIConGPU
 *
 * @param argc count of arguments in argv
 * @param argv arguments of program start
 */

#define CUPTI_CALL(call)                                                \                             
  do {                                                                  \                             
    CUptiResult _status = call;                                         \                             
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \                             
      cuptiGetResultString(_status, &errstr);                           \                             
      std::cerr << __FILE__ << ":" << __LINE__ << ":" << "error: function " << #call  << "failed with error " << errstr << std::endl; \
                       \
      std::exit(-1);                                                         \                        
    }                                                                   \                             
  } while (0)

#define BUF_SIZE (32 * 1024 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \                      
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

void CUPTIAPI take_buffer(uint8_t **buffer, size_t *size, size_t *max_num_records)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    std::cerr << "Error: out of memory" << std::endl;
    std::exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *max_num_records = 0;
}

long g_last_end = 0, g_last_end_api = 0; 
int g_rank = -1;
std::ofstream g_ofs, g_ofs_api;

void CUPTIAPI return_buffer(CUcontext ctx, uint32_t stream_id, uint8_t *buffer, size_t size, size_t valid_size)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;
  int num_kernel_records=0, num_api_records=0;
  if (valid_size > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, valid_size, &record);
      if (status == CUPTI_SUCCESS) {
         //num_records++;
         if((record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)||(record->kind == CUPTI_ACTIVITY_KIND_KERNEL))
         { 
            num_kernel_records++;
            CUpti_ActivityKernel2 *kernel = (CUpti_ActivityKernel2 *) record;
            //if(g_last_end==0)
            //{
               //g_last_end = kernel->start; //kernel->start;
            //}
	    g_ofs << kernel->start << " " << kernel->end << " " << kernel->correlationId << /*" " << kernel->end - kernel->start << " " << kernel->start - g_last_end <<*/ std::endl;
            /*
            if((long)(kernel->start) >= g_last_end){
               g_ofs << kernel->start << " " << kernel->end << " " << kernel->correlationId << " " << kernel->end - kernel->start << " " << kernel->start - g_last_end << std::endl;
               g_last_end = kernel->end;
            }
            else if((long)(kernel->end) >= g_last_end)
            {
               g_last_end = kernel->end;
            }*/
         }
         if((record->kind == CUPTI_ACTIVITY_KIND_RUNTIME)||(record->kind == CUPTI_ACTIVITY_KIND_DRIVER))
         {
             num_api_records++;
             CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
             /*if(g_last_end_api==0)
             {
                g_last_end_api = api->start;
             }*/
             //g_ofs_api << api->start << " " << api->correlationId << std::endl;
         }
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);
    //g_ofs << "dump " << num_kernel_records << std::endl;
    //g_ofs_api << "dump " << num_api_records << std::endl;
    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, stream_id, &dropped));
    if (dropped != 0) {
      std::cerr << "Dropped " << (unsigned int) dropped << "activity records" << std::endl;
    }

  }

  free(buffer);
}

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

int sockpull_t, sockpush_t, sockpull_p, sockpush_p;

void* some_thrust_method(void *data)
{
    /*pthread_mutex_lock(&mutex1);
    while(!P_to_T)
    {
      pthread_cond_wait(&condition1, &mutex1);
    }
    std::cout << "Thrust received signal\n";
    P_to_T = false;
    pthread_mutex_unlock(&mutex1);

    pthread_mutex_lock(&mutex2);
    std::cout << "Thrust sending signal\n";
    T_to_P = true;
    pthread_mutex_unlock(&mutex2);
    pthread_cond_signal(&condition2);*/
    sockpull_t = nn_socket (AF_SP, NN_PULL);
    assert (sockpull_t >= 0);
    std::cerr << "[t]t_to_p socket ok" << std::endl;
    assert (nn_bind (sockpull_t, "ipc:///tmp/t_to_p.ipc") >= 0);
    std::cerr << "t_to_p bind ok" << std::endl;
    sockpush_t = nn_socket (AF_SP, NN_PUSH);
    assert (sockpush_t >= 0);
    std::cerr << "[t]p_to_t socket ok" << std::endl;
    assert (nn_connect (sockpush_t, "ipc:///tmp/p_to_t.ipc") >= 0);
    std::cerr << "p_to_t connect ok" << std::endl;

    while(1)
    {
    while (1)
    {
      char *buf = NULL;
      int bytes = nn_recv (sockpull_t, &buf, NN_MSG, 0);
      assert (bytes >= 0);
      std::cerr << "[T] : " << buf << std::endl;
      nn_freemsg (buf);
      break;
    } 
    
    const char * msg = "T to P";
    int bytes = nn_send (sockpush_t, msg, strlen (msg) + 1, 0);
    assert (bytes == strlen (msg) + 1);
    std::cerr << "[T] T to P" << std::endl;
    }
    return NULL;
}

MPI_Comm MPI_COMM_WORLD_INSITU;
cudaIpcMemHandle_t *g_mem_handle=NULL;

int main(int argc, char **argv)
{
    MPI_CHECK(MPI_Init(&argc, &argv));

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
    //g_mem_handle = (cudaIpcMemHandle_t *)mmap(NULL, sizeof(cudaIpcMemHandle_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);

    //MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, g_rank%2, g_rank, &MPI_COMM_WORLD_INSITU));
    pthread_t some_thrust_method_handle;
    if(pthread_create(&some_thrust_method_handle, NULL, some_thrust_method, NULL))
    {
       std::cout << "Error calling pthread_create" << std::endl;
    }
    sockpush_p = nn_socket (AF_SP, NN_PUSH);
    assert (sockpush_p >= 0);
    std::cerr << "[p]t_to_p socket ok" << std::endl;
    assert (nn_connect (sockpush_p, "ipc:///tmp/t_to_p.ipc") >= 0);
    std::cerr << "t_to_p connect ok" << std::endl;
    sockpull_p = nn_socket (AF_SP, NN_PULL);
    assert (sockpull_p >= 0);
    std::cerr << "[p]p_to_t socket ok" << std::endl;
    assert (nn_bind (sockpull_p, "ipc:///tmp/p_to_t.ipc") >= 0);
    std::cerr << "p_to_t bind ok" << std::endl;

    int insitu_rank=-1;
    //MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &insitu_rank));

    //if(g_rank % 2 == 0)
    //{
    std::string filename_k(patch::to_string(g_rank)+"_kernel_picongpu.out");
    std::string filename_rd(patch::to_string(g_rank)+"_api_picongpu.out");
    g_ofs.open(filename_k.c_str());
    g_ofs_api.open(filename_rd.c_str());
    //}
    //else
    //{
    /*
    std::string filename_k(patch::to_string(g_rank)+"_kernel_thrust.out");
    std::string filename_rd(patch::to_string(g_rank)+"_api_thrust.out");
    g_ofs.open(filename_k.c_str());
    g_ofs_api.open(filename_rd.c_str());
    */
    //}
    cuptiActivityRegisterCallbacks(take_buffer, return_buffer);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    //cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

    int errorCode = 1;

    //if(g_rank % 2 == 0)
    //{
        //std::cout << "picongpu " << insitu_rank << " " << g_rank << std::endl;
    picongpu::simulation_starter::SimStarter sim;
    ArgsParser::ArgsErrorCode parserCode = sim.parseConfigs(argc, argv);

    /*if (!sim.parseConfigs(argc, argv))
    {
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    sim.load();
    sim.start();
    sim.unload();*/

    switch(parserCode)
    {
        case ArgsParser::ERROR:
            errorCode = 1;
            break;
        case ArgsParser::SUCCESS:
            sim.load();
            sim.start();
            sim.unload();
            /*set error code to valid (1) after the simulation terminates*/
        case ArgsParser::SUCCESS_EXIT:
            errorCode = 0;
            break;
    };
    //}
    //else
    //{
        //thrust
        /*
        std::cout << "thrust " << insitu_rank << " " << g_rank << std::endl;
        while(g_mem_handle == NULL){}
        std::cerr << "main::cudaIpcOpenMemHandle mem_handle " << g_mem_handle << " " << (*g_mem_handle) << std::endl;
        void *d_ptr=NULL;
        cudaError_t err = cudaIpcOpenMemHandle((void **) &d_ptr, *g_mem_handle, cudaIpcMemLazyEnablePeerAccess);
        if(err != cudaSuccess)
        {
            std::cerr << err << " cudaErrorMapBufferObjectFailed-" << cudaErrorMapBufferObjectFailed << " cudaErrorInvalidResourceHandle-" << cudaErrorInvalidResourceHandle << " cudaErrorTooManyPeers-" << cudaErrorTooManyPeers << std::endl;
        }
        else
        {
            std::cerr << "main::cudaIpcOpenMemHandle d_ptr " << d_ptr << std::endl;
        }
        */
    //}

    //cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
    //cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);
    //cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

    /*if(pthread_join(some_thrust_method_handle, NULL))
    {
        std::cout << "Error in pthread_join" << std::endl;
    }*/

    MPI_CHECK(MPI_Finalize());

    return 0;
}
