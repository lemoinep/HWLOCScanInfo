
#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunknown-attributes"


#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>

  
//Link HIP
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hipblas.h"
#include "hipsolver.h"
#include "hipblas-export.h"

//Link Specx
#include "SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"
#include "Utils/SpTimer.hpp"
#include "Utils/small_vector.hpp"
#include "Utils/SpConsumerThread.hpp"
#include "SpComputeEngine.hpp"
#include "Speculation/SpSpeculativeModel.hpp"
   

//Link for hwloc
#include "hwloc.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <assert.h>

//Link for likwid
//#include <likwid.h>
#include "likwid.h"
#include <signal.h>

//CMD: hipconfig --full


//==============================================================================================


static int sleeptime = 1;

static int run = 1;

void  INThandler(int sig)
{
    signal(sig, SIG_IGN);
    run = 0;
}


//==============================================================================================

static void print_children(hwloc_topology_t topology, hwloc_obj_t obj,
                           int depth)
{
    char type[32], attr[1024];
    unsigned i;

    hwloc_obj_type_snprintf(type, sizeof(type), obj, 0);
    printf("%*s%s", 2*depth, "", type);
    if (obj->os_index != (unsigned) -1)
      printf("#%u", obj->os_index);
    hwloc_obj_attr_snprintf(attr, sizeof(attr), obj, " ", 0);
    if (*attr)
      printf("(%s)", attr);
    printf("\n");
    for (i = 0; i < obj->arity; i++) {
        print_children(topology, obj->children[i], depth + 1);
    }
}

void getInfoSystem()
{
    int depth;
    unsigned i, n;
    unsigned long size;
    int levels;
    char string[128];
    int topodepth;
    void *m;
    hwloc_topology_t topology;
    hwloc_cpuset_t cpuset;
    hwloc_obj_t obj;

    /* Allocate and initialize topology object. */
    hwloc_topology_init(&topology);

    /* ... Optionally, put detection configuration here to ignore
       some objects types, define a synthetic topology, etc....

       The default is to detect all the objects of the machine that
       the caller is allowed to access.  See Configure Topology
       Detection. */

    /* Perform the topology detection. */
    hwloc_topology_load(topology);

    /* Optionally, get some additional topology information
       in case we need the topology depth later. */
    topodepth = hwloc_topology_get_depth(topology);

    printf("-------------------------------------------------------------------\n");
    std::cout << "[INFO]: TOPOLOGY SYSTEM"<<"\n";

    /*****************************************************************
     * First example:
     * Walk the topology with an array style, from level 0 (always
     * the system level) to the lowest level (always the proc level).
     *****************************************************************/
    for (depth = 0; depth < topodepth; depth++) {
        printf(" *** Objects at level %d\n", depth);
        for (i = 0; i < hwloc_get_nbobjs_by_depth(topology, depth);
             i++) {
            hwloc_obj_type_snprintf(string, sizeof(string),
				    hwloc_get_obj_by_depth(topology, depth, i), 0);
            printf("[%u]:%s  ", i, string);
            if (i%10==0) printf("\n");
        }

        printf("\n\n");
    }

    printf("-------------------------------------------------------------------\n");

    /*****************************************************************
     * Second example:
     * Walk the topology with a tree style.
     *****************************************************************/
    printf("[INFO]: Printing overall tree\n");
    print_children(topology, hwloc_get_root_obj(topology), 0);

    printf("-------------------------------------------------------------------\n");
    /*****************************************************************
     * Third example:
     * Print the number of packages.
     *****************************************************************/
    depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);
    if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
        printf("[INFO]: The number of packages is unknown\n");
    } else {
        printf("[INFO]: %u package(s)\n",
               hwloc_get_nbobjs_by_depth(topology, depth));
    }
    printf("-------------------------------------------------------------------\n");
    /*****************************************************************
     * Fourth example:
     * Compute the amount of cache that the first logical processor
     * has above it.
     *****************************************************************/
    levels = 0;
    size = 0;
    for (obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, 0);
         obj;
         obj = obj->parent)
      if (hwloc_obj_type_is_cache(obj->type)) {
        levels++;
        size += obj->attr->cache.size;
      }
    printf("[INFO]:  Logical processor 0 has %d caches totaling %luKB\n",levels, size / 1024);
    printf("-------------------------------------------------------------------\n");

    /*****************************************************************
     * Fifth example:
     * Bind to only one thread of the last core of the machine.
     *
     * First find out where cores are, or else smaller sets of CPUs if
     * the OS doesn't have the notion of a "core".
     *****************************************************************/
    depth = hwloc_get_type_or_below_depth(topology, HWLOC_OBJ_CORE);

    /* Get last core. */
    obj = hwloc_get_obj_by_depth(topology, depth,hwloc_get_nbobjs_by_depth(topology, depth) - 1);
    if (obj) {
        /* Get a copy of its cpuset that we may modify. */
        cpuset = hwloc_bitmap_dup(obj->cpuset);

        /* Get only one logical processor (in case the core is
           SMT/hyper-threaded). */
        hwloc_bitmap_singlify(cpuset);

        /* And try to bind ourself there. */
        if (hwloc_set_cpubind(topology, cpuset, 0)) {
            char *str;
            int error = errno;
            hwloc_bitmap_asprintf(&str, obj->cpuset);
            printf("Couldn't bind to cpuset %s: %s\n", str, strerror(error));
            free(str);
        }

        /* Free our cpuset copy */
        hwloc_bitmap_free(cpuset);
    }


    printf("-------------------------------------------------------------------\n");

    /*****************************************************************
     * Sixth example:
     * Allocate some memory on the last NUMA node, bind some existing
     * memory to the last NUMA node.
     *****************************************************************/
    /* Get last node. There's always at least one. */
    n = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
    obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, n - 1);

    size = 1024*1024;
    m = hwloc_alloc_membind(topology, size, obj->nodeset,
                            HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
    hwloc_free(topology, m, size);

    m = malloc(size);
    hwloc_set_area_membind(topology, m, size, obj->nodeset,HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
    free(m);

    printf("-------------------------------------------------------------------\n");

    /* Destroy topology object. */
    hwloc_topology_destroy(topology);

}


void getInfoMemory()
{
    hwloc_topology_t topology;
    hwloc_obj_t core, *nodes, bestnode;
    struct hwloc_location initiator;
    unsigned i,n;
    char *s, *buffer;
    int err;

    /* Allocate, initialize and load topology object. */
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    /* Find max number of NUMA nodes to allocate the array for hwloc_get_local_numanode_objs() */
    n = hwloc_bitmap_weight(hwloc_topology_get_topology_nodeset(topology));
    printf("There are %u NUMA nodes\n", n);

    //nodes = malloc(n * sizeof(*nodes));  <<Error allocation
    assert(nodes);

    /* Take the first core */

    
    core = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, 0);
    if (!core)
      goto out;

    hwloc_bitmap_asprintf(&s, core->cpuset);
    printf("Core L#0 cpuset = %s\n", s);
    free(s);
    

    /* setup the initiator to the first core cpuset */

    
    initiator.type = HWLOC_LOCATION_TYPE_CPUSET;
    initiator.location.cpuset = core->cpuset;
    

    /* get local NUMA nodes and display their attributes */

    
    err = hwloc_get_local_numanode_objs(topology, &initiator, &n, nodes,HWLOC_LOCAL_NUMANODE_FLAG_LARGER_LOCALITY);
    printf("Found %u local NUMA nodes\n", n);
    for(i=0; i<n; i++) {
      hwloc_uint64_t latency, bandwidth;

      printf("NUMA node L#%u P#%u (subtype %s) is local to core L#0\n", nodes[i]->logical_index, nodes[i]->os_index, nodes[i]->subtype);

      err = hwloc_memattr_get_value(topology, HWLOC_MEMATTR_ID_BANDWIDTH, nodes[i], &initiator, 0, &bandwidth);
      if (err < 0) {
        printf("  bandwidth is unknown\n");
      } else {
        printf("  bandwidth = %llu MiB/s\n", (unsigned long long) bandwidth);
      }
      err = hwloc_memattr_get_value(topology, HWLOC_MEMATTR_ID_LATENCY, nodes[i], &initiator, 0, &latency);
      if (err < 0) {
        printf("  latency is unknown\n");
      } else {
        printf("  latency = %llu ns\n", (unsigned long long) latency);
      }
    }
    free(nodes);
    
    /* allocate on best-bandwidth node */
    err = hwloc_memattr_get_best_target(topology, HWLOC_MEMATTR_ID_BANDWIDTH, &initiator, 0, &bestnode, NULL);
    if (err < 0) {
      printf("Couldn't find best NUMA node for bandwidth to core L#0\n");
    } else {
      printf("Best bandwidth NUMA node for core L#0 is L#%u P#%u\n", bestnode->logical_index, bestnode->os_index);
      //buffer = hwloc_alloc_membind(topology, 1048576, bestnode->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
      //printf("Allocated buffer %p on best node\n", buffer);
      free(buffer);
    }
    

    out:
        /* Destroy topology object. */
        hwloc_topology_destroy(topology);

    
}


int getNodeset()
{
    hwloc_topology_t topology;
    hwloc_bitmap_t set;
    hwloc_const_bitmap_t cset;
    hwloc_membind_policy_t policy;
    const struct hwloc_topology_support *support;
    int nbnodes;
    hwloc_obj_t obj;
    char *buffer, *s;
    unsigned i;
    int err;

    /* create a topology */
    err = hwloc_topology_init(&topology);
    if (err < 0) {
        fprintf(stderr, "failed to initialize the topology\n");
        return EXIT_FAILURE;
    }
    err = hwloc_topology_load(topology);
    if (err < 0) {
        fprintf(stderr, "failed to load the topology\n");
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }

    /* retrieve the entire set of NUMA nodes and count them */
    cset = hwloc_topology_get_topology_nodeset(topology);
    nbnodes = hwloc_bitmap_weight(cset);
    /* there's always at least one NUMA node */
    assert(nbnodes > 0);
    printf("[INFO]: There are %d nodes in the machine\n", nbnodes);

    /* get the process memory binding as a nodeset */
    set = hwloc_bitmap_alloc();
    if (!set) {
        fprintf(stderr, "failed to allocate a bitmap\n");
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }
    err = hwloc_get_membind(topology, set, &policy, HWLOC_MEMBIND_BYNODESET);
    if (err < 0) {
        fprintf(stderr, "failed to retrieve my memory binding and policy\n");
        hwloc_topology_destroy(topology);
        hwloc_bitmap_free(set);
        return EXIT_FAILURE;
    }

    /* print the corresponding NUMA nodes */
    hwloc_bitmap_asprintf(&s, set);
    printf("[INFO]: Bound to nodeset %s with contains:\n", s);
    free(s);
    hwloc_bitmap_foreach_begin(i, set) {
        obj = hwloc_get_numanode_obj_by_os_index(topology, i);
        printf("  node #%u (OS index %u) with %llu bytes of memory\n",
        obj->logical_index, i, (unsigned long long) obj->attr->numanode.local_memory);
    } hwloc_bitmap_foreach_end();
    hwloc_bitmap_free(set);

    /* check alloc+bind support */
    support = hwloc_topology_get_support(topology);
    if (support->membind->bind_membind) {
        printf("[INFO]: BIND memory binding policy is supported\n");
    } else {
        printf("[INFO]: BIND memory binding policy is NOT supported\n");
    }
    if (support->membind->alloc_membind) {
        printf("[INFO]: Allocating bound memory is supported\n");
    } else {
        printf("[INFO]: Allocating bound memory is NOT supported\n");
    }

    /* allocate memory of each nodes */
    /*
    printf("allocating memory on each node\n");
    obj = NULL;
    buffer = NULL;
    while ((obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, obj)) != NULL) {
        buffer = hwloc_alloc_membind(topology, 4096, obj->nodeset, HWLOC_MEMBIND_BIND,
                                    HWLOC_MEMBIND_STRICT|HWLOC_MEMBIND_BYNODESET);
        if (!buffer) {
        fprintf(stderr, "failed to allocate memory on node %u\n", obj->os_index);
        hwloc_topology_destroy(topology);
        return EXIT_SUCCESS;
        }
    }
    */

    /* check where buffer is allocated */
    /*
    set = hwloc_bitmap_alloc();
    if (!set) {
        fprintf(stderr, "failed to allocate a bitmap\n");
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }
    err = hwloc_get_area_membind(topology, buffer, 4096, set, &policy, HWLOC_MEMBIND_BYNODESET);
    if (err < 0) {
        fprintf(stderr, "failed to retrieve the buffer binding and policy\n");
        hwloc_topology_destroy(topology);
        hwloc_bitmap_free(set);
        return EXIT_FAILURE;
    }
    */

    /* check the binding policy, it should be what we requested above,
    * but may be different if the implementation of different policies
    * is identical for the current operating system (e.g. if BIND is the DEFAULT).
    */
    //printf("buffer membind policy is %d while we requested %d\n",policy, HWLOC_MEMBIND_BIND);

    /* print the corresponding NUMA nodes */
    /*
    hwloc_bitmap_asprintf(&s, set);
    printf("buffer bound to nodeset %s with contains:\n", s);
    free(s);
    hwloc_bitmap_foreach_begin(i, set) {
        obj = hwloc_get_numanode_obj_by_os_index(topology, i);
        printf("  node #%u (OS index %u) with %llu bytes of memory\n",
        obj->logical_index, i, (unsigned long long) obj->attr->numanode.local_memory);
    } hwloc_bitmap_foreach_end();
    hwloc_bitmap_free(set);
    */

    /* try to migrate the buffer to the first node */
    obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, 0);
    err = hwloc_set_area_membind(topology, buffer, 4096, obj->nodeset, HWLOC_MEMBIND_BIND,
                                HWLOC_MEMBIND_MIGRATE|HWLOC_MEMBIND_BYNODESET);
    if (err < 0) {
        fprintf(stderr, "failed to migrate buffer\n");
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }

    hwloc_topology_destroy(topology);
    return EXIT_SUCCESS;
}


void getInfoGPU()
{
    hwloc_topology_t topology;
    hwloc_obj_t obj;
    unsigned n, i;
    int devid, platformid;
    const char *dev;

    /* Allocate, initialize and load topology object. */
    hwloc_topology_init(&topology);
    hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
    hwloc_topology_load(topology);

    /* Find CUDA devices through the corresponding OS devices */
    n = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_OS_DEVICE);

    for (i = 0; i < n ; i++) {
        printf("-------------------------------------------------------------------\n");
        const char *s;
        obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_OS_DEVICE, i);
        printf("[INFO]: %s:\n", obj->name);

        /* obj->attr->osdev.type is HWLOC_OBJ_OSDEV_COPROC|HWLOC_OBJ_OSDEV_GPU */

        s = hwloc_obj_get_info_by_name(obj, "Backend");
        /* obj->subtype also contains CUDA or OpenCL since v2.0 */

        printf("[INFO]: %s\n",s);

        if (s && !strcmp(s, "CUDA")) {
            /* This is a CUDA device */
            assert(!strncmp(obj->name, "cuda", 4));
            devid = atoi(obj->name + 4);
            printf("[INFO]: CUDA device %d\n", devid);

            s = hwloc_obj_get_info_by_name(obj, "GPUModel");
            if (s)
            printf("[INFO]: Model: %s\n", s);

            s = hwloc_obj_get_info_by_name(obj, "CUDAGlobalMemorySize");
            if (s)
            printf("[INFO]: Memory: %s\n", s);

            s = hwloc_obj_get_info_by_name(obj, "CUDAMultiProcessors");
            if (s)
            {
            int mp = atoi(s);
            s = hwloc_obj_get_info_by_name(obj, "CUDACoresPerMP");
            if (s) {
                int mp_cores = atoi(s);
                printf("[INFO]: Cores: %d\n", mp * mp_cores);
            }
            }
        }

        if (s && !strcmp(s, "OpenCL")) {
            /* This is an OpenCL device */
            assert(!strncmp(obj->name, "opencl", 6));
            platformid = atoi(obj->name + 6);
            printf("[INFO]: OpenCL platform %d\n", platformid);
            dev = strchr(obj->name + 6, 'd');
            devid = atoi(dev + 1);
            printf("[INFO]: OpenCL device %d\n", devid);

            s = hwloc_obj_get_info_by_name(obj, "GPUModel");
            if (s)
            printf("[INFO]: Model: %s\n", s);

            s = hwloc_obj_get_info_by_name(obj, "OpenCLGlobalMemorySize");
            if (s)
            printf("[INFO]: Memory: %s\n", s);
        }

        /* One can also use helpers from hwloc/cuda.h, hwloc/cudart.h,
        * hwloc/opencl.h */


        /* Find out cpuset this is connected to */
        while (obj && (!obj->cpuset || hwloc_bitmap_iszero(obj->cpuset)))
            obj = obj->parent;

        if (obj) {
            char *cpuset_string;
            char name[16];
            hwloc_obj_type_snprintf(name, sizeof(name), obj, 0);
            hwloc_bitmap_asprintf(&cpuset_string, obj->cpuset);
            printf("[INFO]: Location: %s P#%u\n", name, obj->os_index);
            printf("[INFO]: Cpuset: %s\n", cpuset_string);
        }
        printf("-------------------------------------------------------------------\n");
    }

    /* Destroy topology object. */
    hwloc_topology_destroy(topology);
}


void getknlmodes()
{
    hwloc_topology_t topology;
	hwloc_obj_t root;
	const char *cluster_mode;
	const char *memory_mode;

	hwloc_topology_init(&topology);
	hwloc_topology_load(topology);

	root = hwloc_get_root_obj(topology);

	cluster_mode = hwloc_obj_get_info_by_name(root, "ClusterMode");
	memory_mode = hwloc_obj_get_info_by_name(root, "MemoryMode");

	printf ("[INFO]: ClusterMode is '%s' MemoryMode is '%s'\n",
		cluster_mode ? cluster_mode : "NULL",
		memory_mode ? memory_mode : "NULL");

	hwloc_topology_destroy(topology);
}


int getCPUsetAndBind()
{
    hwloc_topology_t topology;
    hwloc_bitmap_t set, set2;
    hwloc_const_bitmap_t cset_available, cset_all;
    hwloc_obj_t obj;
    char *buffer;
    char type[64];
    unsigned i;
    int err;

    /* create a topology */
    err = hwloc_topology_init(&topology);
    if (err < 0) {
        fprintf(stderr, "failed to initialize the topology\n");
        return EXIT_FAILURE;
    }
    err = hwloc_topology_load(topology);
    if (err < 0) {
        fprintf(stderr, "failed to load the topology\n");
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }

    /* retrieve the entire set of available PUs */
    cset_available = hwloc_topology_get_topology_cpuset(topology);

    /* retrieve the CPU binding of the current entire process */
    set = hwloc_bitmap_alloc();
    if (!set) {
        fprintf(stderr, "failed to allocate a bitmap\n");
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }
    err = hwloc_get_cpubind(topology, set, HWLOC_CPUBIND_PROCESS);
    if (err < 0) {
        fprintf(stderr, "failed to get cpu binding\n");
        hwloc_bitmap_free(set);
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }

    /* display the processing units that cannot be used by this process */
    if (hwloc_bitmap_isequal(set, cset_available)) {
        printf("[INFO]: This process can use all available processing units in the system\n");
    } else {
        /* compute the set where we currently cannot run.
        * we can't modify cset_available because it's a system read-only one,
        * so we do   set = available &~ set
        */
        hwloc_bitmap_andnot(set, cset_available, set);
        hwloc_bitmap_asprintf(&buffer, set);
        printf("[INFO]: Process cannot use %d process units (%s) among %d in the system\n",
        hwloc_bitmap_weight(set), buffer, hwloc_bitmap_weight(cset_available));
        free(buffer);
        /* restore set where it was before the &~ operation above */
        hwloc_bitmap_andnot(set, cset_available, set);
    }
    /* print the smallest object covering the current process binding */
    obj = hwloc_get_obj_covering_cpuset(topology, set);
    hwloc_obj_type_snprintf(type, sizeof(type), obj, 0);
    printf("[INFO]: Process is bound within object %s logical index %u\n", type, obj->logical_index);

    /* retrieve the single PU where the current thread actually runs within this process binding */
    set2 = hwloc_bitmap_alloc();
    if (!set2) {
        fprintf(stderr, "failed to allocate a bitmap\n");
        hwloc_bitmap_free(set);
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }
    err = hwloc_get_last_cpu_location(topology, set2, HWLOC_CPUBIND_THREAD);
    if (err < 0) {
        fprintf(stderr, "failed to get last cpu location\n");
        hwloc_bitmap_free(set);
        hwloc_bitmap_free(set2);
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }
    /* sanity checks that are not actually needed but help the reader */
    /* this thread runs within the process binding */
    assert(hwloc_bitmap_isincluded(set2, set));
    /* this thread runs on a single PU at a time */
    assert(hwloc_bitmap_weight(set2) == 1);

    /* print the logical number of the PU where that thread runs */
    /* extract the PU OS index from the bitmap */
    i = hwloc_bitmap_first(set2);
    obj = hwloc_get_pu_obj_by_os_index(topology, i);
    printf("[INFO]: Thread is now running on PU logical index %u (OS/physical index %u)\n",
        obj->logical_index, i);

    /* migrate this single thread to where other PUs within the current binding */
    hwloc_bitmap_andnot(set2, set, set2);
    err = hwloc_set_cpubind(topology, set2, HWLOC_CPUBIND_THREAD);
    if (err < 0) {
        fprintf(stderr, "failed to set thread binding\n");
        hwloc_bitmap_free(set);
        hwloc_bitmap_free(set2);
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }
    /* reprint the PU where that thread runs */
    err = hwloc_get_last_cpu_location(topology, set2, HWLOC_CPUBIND_THREAD);
    if (err < 0) {
        fprintf(stderr, "failed to get last cpu location\n");
        hwloc_bitmap_free(set);
        hwloc_bitmap_free(set2);
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }
    /* print the logical number of the PU where that thread runs */
    /* extract the PU OS index from the bitmap */
    i = hwloc_bitmap_first(set2);
    obj = hwloc_get_pu_obj_by_os_index(topology, i);
    printf("[INFO]: Thread is running on PU logical index %u (OS/physical index %u)\n",
        obj->logical_index, i);

    hwloc_bitmap_free(set);
    hwloc_bitmap_free(set2);

    /* retrieve the entire set of all PUs */
    cset_all = hwloc_topology_get_complete_cpuset(topology);
    if (hwloc_bitmap_isequal(cset_all, cset_available)) {
        printf("[INFO]: All hardware PUs are available\n");
    } else {
        printf("[INFO]: Only %d hardware PUs are available in the machine among %d\n",
        hwloc_bitmap_weight(cset_available), hwloc_bitmap_weight(cset_all));
    }

    hwloc_topology_destroy(topology);
    return EXIT_SUCCESS;
}



int getSharedCaches(int argc, char *argv[])
{
    pid_t hispid;
    hwloc_topology_t topology;
    hwloc_bitmap_t set, hisset;
    hwloc_obj_t obj;
    int err;

    /* find the pid of the other process, otherwise use my own pid */
    if (argc >= 2) {
        hispid = atoi(argv[1]);
    } else {
        hispid = getpid();
    }

    /* create a topology with instruction caches enables */
    err = hwloc_topology_init(&topology);
    if (err < 0) {
        fprintf(stderr, "failed to initialize the topology\n");
        return EXIT_FAILURE;
    }
    hwloc_topology_set_icache_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_ALL);
    err = hwloc_topology_load(topology);
    if (err < 0) {
        fprintf(stderr, "failed to load the topology\n");
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }

    /* find where I am running */
    set = hwloc_bitmap_alloc();
    if (!set) {
        fprintf(stderr, "failed to allocate my bitmap\n");
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }
    err = hwloc_get_cpubind(topology, set, 0);
    if (err < 0) {
        fprintf(stderr, "failed to get my binding\n");
        hwloc_bitmap_free(set);
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }

    /* find where the other process is running */
    hisset = hwloc_bitmap_alloc();
    if (!hisset) {
        fprintf(stderr, "failed to allocate his bitmap\n");
        hwloc_bitmap_free(set);
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }
    /* FIXME: on windows, hispid should be replaced with OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, hispid); */
    err = hwloc_get_proc_cpubind(topology, hispid, hisset, 0);
    if (err < 0) {
        fprintf(stderr, "failed to get his binding\n");
        hwloc_bitmap_free(hisset);
        hwloc_bitmap_free(set);
        hwloc_topology_destroy(topology);
        return EXIT_FAILURE;
    }

    /* merge both process binding into mine */
    hwloc_bitmap_or(set, set, hisset);

    /* find the smallest object covering this set */
    obj = hwloc_get_obj_covering_cpuset(topology, set);

    /* display parents of type cache */
    while (obj) {
        if (hwloc_obj_type_is_cache(obj->type)) {
        char type[64];
        char attr[64];
        hwloc_obj_type_snprintf(type, sizeof(type), obj, 0);
        hwloc_obj_attr_snprintf(attr, sizeof(attr), obj, ", ", 0);
        printf("Found object %s with attributes %s\n", type, attr);
        }
        /* next parent up in the tree */
        obj = obj->parent;
    }

    hwloc_bitmap_free(hisset);
    hwloc_bitmap_free(set);
    hwloc_topology_destroy(topology);
    return EXIT_SUCCESS;
}


//==============================================================================================

// https://github.com/RRZE-HPC/likwid/wiki/LikwidAPI-and-MarkerAPI

int testLikwid()
{
    int i, j;
    int err;
    int* cpus;
    int gid;
    double result = 0.0;
    char estr[] = "L2_LINES_IN_ALL:PMC0,L2_TRANS_L2_WB:PMC1";
    char* enames[2] = {"L2_LINES_IN_ALL:PMC0","L2_TRANS_L2_WB:PMC1"};
    int n = sizeof(enames) / sizeof(enames[0]);
    //perfmon_setVerbosity(3);
    // Load the topology module and print some values.
    err = topology_init();
    if (err < 0)
    {
        printf("Failed to initialize LIKWID's topology module\n");
        return 1;
    }
    // CpuInfo_t contains global information like name, CPU family, ...
    CpuInfo_t info = get_cpuInfo();
    // CpuTopology_t contains information about the topology of the CPUs.
    CpuTopology_t topo = get_cpuTopology();
    // Create affinity domains. Commonly only needed when reading Uncore counters
    affinity_init();

    printf("Likwid example on a %s with %d CPUs\n", info->name, topo->numHWThreads);

    cpus = (int*)malloc(topo->numHWThreads * sizeof(int));
    if (!cpus)
        return 1;

    for (i=0;i<topo->numHWThreads;i++)
    {
        cpus[i] = topo->threadPool[i].apicId;
    }

    // Must be called before perfmon_init() but only if you want to use another
    // access mode as the pre-configured one. For direct access (0) you have to
    // be root.
    //accessClient_setaccessmode(0);

    // Initialize the perfmon module.
    err = perfmon_init(topo->numHWThreads, cpus);
    if (err < 0)
    {
        printf("Failed to initialize LIKWID's performance monitoring module\n");
        topology_finalize();
        return 1;
    }

    // Add eventset string to the perfmon module.
    gid = perfmon_addEventSet(estr);
    if (gid < 0)
    {
        printf("Failed to add event string %s to LIKWID's performance monitoring module\n", estr);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }

    // Setup the eventset identified by group ID (gid).
    err = perfmon_setupCounters(gid);
    if (err < 0)
    {
        printf("Failed to setup group %d in LIKWID's performance monitoring module\n", gid);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }
    // Start all counters in the previously set up event set.
    err = perfmon_startCounters();
    if (err < 0)
    {
        printf("Failed to start counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }


    // Perform some work.
    sleep(2);

    // Read and record current event counts.
    err = perfmon_readCounters();
    if (err < 0)
    {
        printf("Failed to read counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }

    // Print the result of every thread/CPU for all events in estr, counting from last read/startCounters().
    printf("Work task 1/2 measurements:\n");
    for (j=0; j<n; j++)
    {
        for (i = 0;i < topo->numHWThreads; i++)
        {
            result = perfmon_getLastResult(gid, j, i);
            printf("- event set %s at CPU %d: %f\n", enames[j], cpus[i], result);
        }
    }


    // Perform another piece of work
    sleep(2);

    // Read and record current event counts.
    err = perfmon_readCounters();
    if (err < 0)
    {
        printf("Failed to read counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }

    // Print the result of every thread/CPU for all events in estr, counting between the 
    // previous two calls of perfmon_readCounters().
    printf("Work task 2/2 measurements:\n");
    for (j=0; j<n; j++)
    {
        for (i = 0;i < topo->numHWThreads; i++)
        {
            result = perfmon_getLastResult(gid, j, i);
            printf("- event set %s at CPU %d: %f\n", enames[j], cpus[i], result);
        }
    }



    // Stop all counters in the currently-active event set.
    err = perfmon_stopCounters();
    if (err < 0)
    {
        printf("Failed to stop counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }

    // Print the result of every thread/CPU for all events in estr, counting since counters first started.
    printf("Total sum measurements:\n");
    for (j=0; j<n; j++)
    {
        for (i = 0;i < topo->numHWThreads; i++)
        {
            result = perfmon_getResult(gid, j, i);
            printf("- event set %s at CPU %d: %f\n", enames[j], cpus[i], result);
        }
    }


    free(cpus);
    // Uninitialize the perfmon module.
    perfmon_finalize();
    affinity_finalize();
    // Uninitialize the topology module.
    topology_finalize();
}


int testLikwid2()
{
    int i, j;
    int err;
    int* cpus;
    int gid;
    double result = 0.0;
    char estr[] = "L2_LINES_IN_ALL:PMC0,L2_TRANS_L2_WB:PMC1";
    //perfmon_setVerbosity(3);
    // Load the topology module and print some values.
    err = topology_init();
    if (err < 0)
    {
        printf("Failed to initialize LIKWID's topology module\n");
        return 1;
    }
    // CpuInfo_t contains global information like name, CPU family, ...
    CpuInfo_t info = get_cpuInfo();
    // CpuTopology_t contains information about the topology of the CPUs.
    CpuTopology_t topo = get_cpuTopology();
    // Create affinity domains. Commonly only needed when reading Uncore counters
    affinity_init();
    printf("Likwid example on a %s with %d CPUs\n", info->name, topo->numHWThreads);
    cpus = (int*)malloc(topo->numHWThreads * sizeof(int));
    if (!cpus)
        return 1;
    for (i=0;i<topo->numHWThreads;i++)
    {
        cpus[i] = topo->threadPool[i].apicId;
    }
    // Must be called before perfmon_init() but only if you want to use another
    // access mode as the pre-configured one. For direct access (0) you have to
    // be root.
    //accessClient_setaccessmode(0);
    // Initialize the perfmon module.
    err = perfmon_init(topo->numHWThreads, cpus);
    if (err < 0)
    {
        printf("Failed to initialize LIKWID's performance monitoring module\n");
        topology_finalize();
        return 1;
    }
    // Add eventset string to the perfmon module.
    gid = perfmon_addEventSet(estr);
    if (gid < 0)
    {
        printf("Failed to add event string %s to LIKWID's performance monitoring module\n", estr);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }
    // Setup the eventset identified by group ID (gid).
    err = perfmon_setupCounters(gid);
    if (err < 0)
    {
        printf("Failed to setup group %d in LIKWID's performance monitoring module\n", gid);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }
    // Start all counters in the previously set up event set.
    err = perfmon_startCounters();
    if (err < 0)
    {
        printf("Failed to start counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }
    // Perform something
    sleep(10);
    // Stop all counters in the previously started event set.
    err = perfmon_stopCounters();
    if (err < 0)
    {
        printf("Failed to stop counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }
    // Print the result of every thread/CPU for all events in estr.
    char* ptr = strtok(estr,",");
    j = 0;
    while (ptr != NULL)
    {
        for (i = 0;i < topo->numHWThreads; i++)
        {
            result = perfmon_getResult(gid, j, i);
            printf("Measurement result for event set %s at CPU %d: %f\n", ptr, cpus[i], result);
        }
        ptr = strtok(NULL,",");
        j++;
    }
    free(cpus);
    // Uninitialize the perfmon module.
    perfmon_finalize();
    affinity_finalize();
    // Uninitialize the topology module.
    topology_finalize();
    return 0;
}


int testLikwidMonitoring()
{
    int i, c, err = 0;
    double timer = 0.0;
    topology_init();
    numa_init();
    affinity_init();
    timer_init();
    CpuInfo_t cpuinfo = get_cpuInfo();
    CpuTopology_t cputopo = get_cpuTopology();
    int numCPUs = cputopo->activeHWThreads;
    //int* cpus = malloc(numCPUs * sizeof(int));
    int *cpus;
    cpus = (int*) malloc(numCPUs * sizeof(int));
    if (!cpus)
    {
        affinity_finalize();
        numa_finalize();
        topology_finalize();
        return 1;
    }
    c = 0;
    for (i=0;i<cputopo->numHWThreads;i++)
    {
        if (cputopo->threadPool[i].inCpuSet)
        {
            cpus[c] = cputopo->threadPool[i].apicId;
            c++;
        }
    }
    NumaTopology_t numa = get_numaTopology();
    AffinityDomains_t affi = get_affinityDomains();
    timer = timer_getCpuClock();
    perfmon_init(numCPUs, cpus);
    int gid1 = perfmon_addEventSet("L2");
    if (gid1 < 0)
    {
        printf("Failed to add performance group L2\n");
        err = 1;
        free(cpus);
        perfmon_finalize();
        affinity_finalize();
        numa_finalize();
        topology_finalize();
        return 0;
    }
    int gid2 = perfmon_addEventSet("L3");
    if (gid2 < 0)
    {
        printf("Failed to add performance group L3\n");
        err = 1;
        free(cpus);
        perfmon_finalize();
        affinity_finalize();
        numa_finalize();
        topology_finalize();
        return 0;
    }
    int gid3 = perfmon_addEventSet("ENERGY");
    if (gid3 < 0)
    {
        printf("Failed to add performance group ENERGY\n");
        err = 1;
        free(cpus);
        perfmon_finalize();
        affinity_finalize();
        numa_finalize();
        topology_finalize();
        return 0;
    }
    signal(SIGINT, INThandler);

    while (run)
    {
        perfmon_setupCounters(gid1);
        perfmon_startCounters();
        sleep(sleeptime);
        perfmon_stopCounters();
        for (c = 0; c < 8; c++)
        {
            for (i = 0; i< perfmon_getNumberOfMetrics(gid1); i++)
            {
                printf("%s,cpu=%d %f\n", perfmon_getMetricName(gid1, i), cpus[c], perfmon_getLastMetric(gid1, i, c));
            }
        }
        perfmon_setupCounters(gid2);
        perfmon_startCounters();
        sleep(sleeptime);
        perfmon_stopCounters();
        for (c = 0; c < 8; c++)
        {
            for (i = 0; i< perfmon_getNumberOfMetrics(gid2); i++)
            {
                printf("%s,cpu=%d %f\n", perfmon_getMetricName(gid2, i), cpus[c], perfmon_getLastMetric(gid2, i, c));
            }
        }
        perfmon_setupCounters(gid3);
        perfmon_startCounters();
        sleep(sleeptime);
        perfmon_stopCounters();
        for (c = 0; c < 8; c++)
        {
            for (i = 0; i< perfmon_getNumberOfMetrics(gid3); i++)
            {
                printf("%s,cpu=%d %f\n", perfmon_getMetricName(gid3, i), cpus[c], perfmon_getLastMetric(gid3, i, c));
            }
        }
    }

    free(cpus);
    perfmon_finalize();
    affinity_finalize();
    numa_finalize();
    topology_finalize();
    return 0;
}

//==============================================================================================


int main(int argc, char *argv[])
{
    bool isLikwidOn=false;
    bool ishwlocOn=true;
    int count, device;
    hipGetDeviceCount(&count);
    hipGetDevice(&device);
    printf("TRIVIAL TEST %d %d \n", device, count);


    if (ishwlocOn)
    {
        printf("===================================================================\n");
        std::cout <<"\n";
        std::cout << "[INFO]: GET INFO SYSTEM"<<"\n";
        getInfoSystem();
        printf("===================================================================\n");


        std::cout <<"\n";
        printf("===================================================================\n");
        std::cout << "[INFO]: GET INFO MEMORY"<<"\n";
        getInfoMemory();
        printf("===================================================================\n");

        std::cout <<"\n";
        printf("===================================================================\n");
        std::cout << "[INFO]: GET INFO GPU"<<"\n";
        getInfoGPU();
        printf("===================================================================\n");

        std::cout <<"\n";
        printf("===================================================================\n");
        std::cout << "[INFO]: GET knlmodes"<<"\n";
        getknlmodes();
        printf("===================================================================\n");

        std::cout <<"\n";  
        printf("===================================================================\n");
        std::cout << "[INFO]: GET CPU set & bind"<<"\n";
        getCPUsetAndBind();
        printf("===================================================================\n");

        std::cout <<"\n";  
        printf("===================================================================\n");
        std::cout << "[INFO]: GET Node Set"<<"\n";
        getNodeset();
        printf("===================================================================\n");


        std::cout <<"\n";  
        printf("===================================================================\n");
        std::cout << "[INFO]: GET Shared Memory"<<"\n";
        getSharedCaches(argc,argv);
        printf("===================================================================\n");
    }



    if (isLikwidOn)
    {
        std::cout <<"\n";  
        printf("===================================================================\n");
        std::cout << "[INFO]: Test Likwid"<<"\n";
        //testLikwid();
        //testLikwid2();
        testLikwidMonitoring();
        printf("===================================================================\n");
    }



    


    std::cout << "[INFO]: WELL DONE :-) FINISHED !"<<"\n";
    return 0;
 
}

