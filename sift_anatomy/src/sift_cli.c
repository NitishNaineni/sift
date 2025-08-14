/*
IPOL SIFT
Copyright (C) 2014, Ives Rey-Otero, CMLA ENS Cachan 
<ives.rey-otero@cmla.ens-cachan.fr>

Version 20140911 (September 11th, 2014)

This C ANSI source code is related to the IPOL publication

    [1] "Anatomy of the SIFT Method." 
        I. Rey Otero  and  M. Delbracio
        Image Processing Online, 2013.
        http://www.ipol.im/pub/algo/rd_anatomy_sift/

An IPOL demo is available at
        http://www.ipol.im/pub/demo/rd_anatomy_sift/





== Patent Warning and License =================================================

The SIFT method is patented 

    [3] "Method and apparatus for identifying scale invariant features
      in an image."
        David G. Lowe
        Patent number: 6711293
        Filing date: Mar 6, 2000
        Issue date: Mar 23, 2004
        Application number: 09/519,89
  
 These source codes are made available for the exclusive aim of serving as
 scientific tool to verify the soundness and completeness of the algorithm
 description. Compilation, execution and redistribution of this file may
 violate patents rights in certain countries. The situation being different
 for every country and changing over time, it is your responsibility to
 determine which patent rights restrictions apply to you before you compile,
 use, modify, or redistribute this file. A patent lawyer is qualified to make
 this determination. If and only if they don't conflict with any patent terms,
 you can benefit from the following license terms attached to this file.


This program is free software: you can use, modify and/or
redistribute it under the terms of the simplified BSD
License. You should have received a copy of this license along
this program. If not, see
<http://www.opensource.org/licenses/bsd-license.html>.

*/
/**
 * @file sift.c
 * @brief [[MAIN]] The SIFT method 
 *
 * @li basic SIFT transform applied to one image
 * @li verbose SIFT transform 
 *
 * @author Ives Rey-Otero <ives.rey-otero@cmla.ens-cachan.fr>
 */



#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>


#include "lib_sift_anatomy.h"
#include "lib_io_scalespace.h"
#include "io_png.h"
#include "lib_util.h"

/* forward declaration from lib_io_scalespace.c */
void dump_scalespace_raw_to_dir(const struct sift_scalespace *ss,
                                const char *out_dir, const char *stem);
/* forward declarations */
static void ensure_dir_exists(const char* dir);
static void dump_raw_extrema_to_dir(const struct sift_keypoints* keys, const char* out_dir);
static void dump_refined_extrema_to_dir(const struct sift_keypoints* keys, const char* out_dir);
static void dump_oriented_keypoints_to_dir(const struct sift_keypoints* keys, const char* out_dir);

#include <string.h>




/* forward declaration for extrema dumping moved above */

void print_usage()
{
    fprintf(stderr, "Anatomy of the SIFT method (www.ipol.im/pub/pre/82/)  ver 20140801         \n");
    fprintf(stderr, "Usage:  sift_cli image [options...]                                        \n");
    fprintf(stderr, "                                                                           \n");
    fprintf(stderr, "   -ss_noct        (8)  number of octaves                                  \n");
    fprintf(stderr, "   -ss_nspo        (3)  number of scales per octaves                       \n");
    fprintf(stderr, "   -ss_dmin      (0.5)  the sampling distance in the first octave          \n");
    fprintf(stderr, "   -ss_smin      (0.8)  blur level on the seed image                       \n");
    fprintf(stderr, "   -ss_sin       (0.5)  assumed level of blur in the input image           \n");
    fprintf(stderr, "                                                                           \n");
    fprintf(stderr, "   -thresh_dog (0.0133) threshold over the DoG response                    \n");
    fprintf(stderr, "   -thresh_edge   (10)  threshold over the ratio of principal curvature    \n");
    fprintf(stderr, "                                                                           \n");
    fprintf(stderr, "   -ori_nbins    (36)   number of bins in the orientation histogram        \n");
    fprintf(stderr, "   -ori_thresh  (0.8)   threhsold for considering local maxima in          \n");
    fprintf(stderr, "                        the orientation histogram                          \n");
    fprintf(stderr, "   -ori_lambda  (1.5)   sets how local is the analysis of the gradient     \n");
    fprintf(stderr, "                        distribution                                       \n");
    fprintf(stderr, "                                                                           \n");
    fprintf(stderr, "   -descr_nhist   (4)   number of histograms per dimension                 \n");
    fprintf(stderr, "   -descr_nori    (8)   number of bins in each histogram                   \n");
    fprintf(stderr, "   -descr_lambda  (6)   sets how local the descriptor is                   \n");
    fprintf(stderr, "                                                                           \n");
    fprintf(stderr, "   --record dir        dump all outputs into a single directory with      \n");
    fprintf(stderr, "                        subfolders: gss, dog, grad_x, grad_y, extrema,     \n");
    fprintf(stderr, "                        c_pre, refined, c_post, edge, border\n");
}


/**
 *
 * Output 
 *   -1 : malformed argument
 *    0 : option not found  
 *    1 : option found
 */
static int pick_option(int* c, char*** v, char* opt, char* val)
{
    int output = 0;
    int argc = *c;
    char **argv = *v;
    // scan the command line for '-opt'
    for(int i = 0; i < argc; i++){
        if (argv[i][0] == '-' && 0 == strcmp(argv[i]+1, opt))
        {
            // check for a corresponding value
            if (i == argc-1){
                output = -1;
            }
            else{
                if (argv[i+1][0] == '-'){
                    output  = -1;
                }
                // if the option call is well formed
                else{
                    // copy the option value ...
                    strcpy(val, argv[i+1]);
                    // ... and remove from the command line
                    for (int j = i; j < argc - 2; j++){
                        (*v)[j] = (*v)[j+2];
                    }
                    *c -= 2;
                    output = 1;
                }
            }
            // print an error if not succes
            if(output == -1){
                fprintf(stderr, "Fatal error: option %s requires an argument.\n", opt);
            }
        }
    }
    return output;
}



static int parse_options(int argc, char** argv,
                         struct sift_parameters* p,
                         char* dump_all_dir)
{
    int isfound;
    char val[128];

    isfound = pick_option(&argc, &argv, "ss_noct", val);
    if (isfound ==  1)    p->n_oct = atoi(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "ss_nspo", val);
    if (isfound ==  1)    p->n_spo = atoi(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "ss_dmin", val);
    if (isfound ==  1)    p->delta_min = atof(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "ss_smin", val);
    if (isfound ==  1)    p->sigma_min = atof(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "ss_sin", val);
    if (isfound ==  1)    p->sigma_in = atof(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "thresh_dog", val);
    if (isfound ==  1)    p->C_DoG = atof(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "thresh_edge", val);
    if (isfound ==  1)    p->C_edge = atof(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "ori_nbins", val);
    if (isfound ==  1)    p->n_bins = atoi(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "ori_thresh", val);
    if (isfound ==  1)    p->t = atof(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "ori_lambda", val);
    if (isfound ==  1)    p->lambda_ori = atof(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "descr_nhist", val);
    if (isfound ==  1)    p->n_hist = atoi(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "descr_nori", val);
    if (isfound ==  1)    p->n_ori = atoi(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "descr_lambda", val);
    if (isfound ==  1)    p->lambda_descr = atof(val);
    if (isfound == -1)    return EXIT_FAILURE;

    isfound = pick_option(&argc, &argv, "-record", val);
    if (isfound == 1) { strcpy(dump_all_dir, val); }
    if (isfound == -1) return EXIT_FAILURE;

    // check for unknown option call
    for(int i = 0; i < argc; i++){
        if (argv[i][0] == '-'){
            fprintf(stderr, "Fatal error: option \"-%s\" is unknown.\n", argv[i]+1);
            print_usage();
            return EXIT_FAILURE;
        }
    }
    // check for input image
    if (argc != 2){
        print_usage();
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}



/** @brief Main SIFT routine
 * 
 * takes one image as input.        
 * outputs the extracted keypoints in the standard output.
 * 
 * @param flag for the SIFT transform
 * 
 * 0 = one image -> one txt file 
 * 1 = one image -> all txt files
 * 2 = one image -> all txt files + scalespace and DoG
 * 
 */
int main(int argc, char **argv)
{
    // Setting default parameters
    struct sift_parameters* p = sift_assign_default_parameters();
    

    // Parsing command line
    char dump_all_dir[FILENAME_MAX] = "";
    int res = parse_options(argc, argv, p, dump_all_dir);
    if (res == EXIT_FAILURE)
        return EXIT_FAILURE;

    /** Loading image */
    size_t w, h;
    float* x = io_png_read_f32_gray(argv[1], &w, &h);
    if(!x)
        fatal_error("File \"%s\" not found.", argv[1]);
    for(int i=0; i < w*h; i++)
        x[i] /= 256.0;

    /** Memory dynamic allocation */
    // WARNING 6 steps of the algorithm are recorded.
    struct sift_keypoints **kk = xmalloc(6*sizeof(struct sift_keypoints*));
    for(int i = 0; i < 6; i++)
        kk[i] = sift_malloc_keypoints();
    // WARNING 4 scale-space representation are recorded (Gaussian, Laplacian, two gradient components)
    struct sift_scalespace **ss = xmalloc(4*sizeof(struct sift_scalespace*));

    /** Algorithm */
    struct sift_keypoints* k = sift_anatomy(x, w, h, p, ss, kk);

    /** OUTPUT */
    /* Keypoint printing removed for GSS/DoG-only mode */

    /* name variable removed */
    /* All keypoint save/debug outputs removed */
    /* Single unified dump directory (only mode supported) */
    if (dump_all_dir[0]) {
        char path[FILENAME_MAX];
        ensure_dir_exists(dump_all_dir);
        /* gss */
        int nw = snprintf(path, sizeof(path), "%s/%s", dump_all_dir, "gss");
        if (nw < 0 || (size_t)nw >= sizeof(path)) fatal_error("Path too long for gss");
        dump_scalespace_raw_to_dir(ss[0], path, "gss");
        /* dog */
        nw = snprintf(path, sizeof(path), "%s/%s", dump_all_dir, "dog");
        if (nw < 0 || (size_t)nw >= sizeof(path)) fatal_error("Path too long for dog");
        dump_scalespace_raw_to_dir(ss[1], path, "dog");
        /* grad_x (dI/dx) */
        nw = snprintf(path, sizeof(path), "%s/%s", dump_all_dir, "grad_x");
        if (nw < 0 || (size_t)nw >= sizeof(path)) fatal_error("Path too long for grad_x");
        /* Note: ensure grad_x directory contains the horizontal derivative */
        dump_scalespace_raw_to_dir(ss[3], path, "grad_x");
        /* grad_y (dI/dy) */
        nw = snprintf(path, sizeof(path), "%s/%s", dump_all_dir, "grad_y");
        if (nw < 0 || (size_t)nw >= sizeof(path)) fatal_error("Path too long for grad_y");
        /* Note: ensure grad_y directory contains the vertical derivative */
        dump_scalespace_raw_to_dir(ss[2], path, "grad_y");
        /* extrema (kA) */
        nw = snprintf(path, sizeof(path), "%s/%s", dump_all_dir, "extrema");
        if (nw < 0 || (size_t)nw >= sizeof(path)) fatal_error("Path too long for extrema");
        dump_raw_extrema_to_dir(kk[0], path);
        /* c_pre (kB) */
        nw = snprintf(path, sizeof(path), "%s/%s", dump_all_dir, "c_pre");
        if (nw < 0 || (size_t)nw >= sizeof(path)) fatal_error("Path too long for c_pre");
        dump_raw_extrema_to_dir(kk[1], path);
        /* refined (kC) */
        nw = snprintf(path, sizeof(path), "%s/%s", dump_all_dir, "refined");
        if (nw < 0 || (size_t)nw >= sizeof(path)) fatal_error("Path too long for refined");
        dump_refined_extrema_to_dir(kk[2], path);
        /* c_post (kD) */
        nw = snprintf(path, sizeof(path), "%s/%s", dump_all_dir, "c_post");
        if (nw < 0 || (size_t)nw >= sizeof(path)) fatal_error("Path too long for c_post");
        dump_raw_extrema_to_dir(kk[3], path);
        /* edge (kE) */
        nw = snprintf(path, sizeof(path), "%s/%s", dump_all_dir, "edge");
        if (nw < 0 || (size_t)nw >= sizeof(path)) fatal_error("Path too long for edge");
        dump_raw_extrema_to_dir(kk[4], path);
        /* border (kF) */
        nw = snprintf(path, sizeof(path), "%s/%s", dump_all_dir, "border");
        if (nw < 0 || (size_t)nw >= sizeof(path)) fatal_error("Path too long for border");
        dump_raw_extrema_to_dir(kk[5], path);
        /* oriented keypoints (k) */
        nw = snprintf(path, sizeof(path), "%s/%s", dump_all_dir, "keys");
        if (nw < 0 || (size_t)nw >= sizeof(path)) fatal_error("Path too long for keys");
        dump_oriented_keypoints_to_dir(k, path);
    }

    

    /* memory deallocation */
    xfree(x);
    xfree(p);
    sift_free_keypoints(k);
    for(int i = 0; i < 6; i++){
        sift_free_keypoints(kk[i]);
    }
    xfree(kk);
    for(int i = 0; i < 4; i++){
        sift_free_scalespace(ss[i]);
    }
    xfree(ss);
    return EXIT_SUCCESS;
}

/* Create directory if it does not exist */
static void ensure_dir_exists(const char* dir)
{
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        if (mkdir(dir, 0777) == -1) {
            /* If it failed for a reason other than already exists, abort */
            if (errno != EEXIST) {
                fatal_error("Failed to create directory %s", dir);
            }
        }
    }
}

static void dump_raw_extrema_to_dir(const struct sift_keypoints* keys, const char* out_dir)
{
    ensure_dir_exists(out_dir);

    char path_int[FILENAME_MAX];
    char path_float[FILENAME_MAX];
    char path_meta[FILENAME_MAX];
    snprintf(path_int, sizeof(path_int), "%s/%s", out_dir, "extrema_int.i32");
    snprintf(path_float, sizeof(path_float), "%s/%s", out_dir, "extrema_float.f32");
    snprintf(path_meta, sizeof(path_meta), "%s/%s", out_dir, "extrema_meta.json");

    /* Prepare contiguous buffers */
    int n = keys->size;
    int (*buf_i)[4] = (int (*)[4])xmalloc((size_t)n * 4 * sizeof(int));
    float (*buf_f)[4] = (float (*)[4])xmalloc((size_t)n * 4 * sizeof(float));

    for (int k = 0; k < n; k++) {
        const struct keypoint* key = keys->list[k];
        /* Ints: (o, s, i, j) */
        buf_i[k][0] = key->o;
        buf_i[k][1] = key->s;
        buf_i[k][2] = key->i;
        buf_i[k][3] = key->j;
        /* Floats: (y_world, x_world, sigma, val) to mirror proto layout */
        /* Note: key->x stores delta*i (y_world), key->y stores delta*j (x_world) */
        buf_f[k][0] = key->x; /* y_world */
        buf_f[k][1] = key->y; /* x_world */
        buf_f[k][2] = key->sigma;
        buf_f[k][3] = key->val;
    }

    /* Write binaries */
    FILE* fi = fopen(path_int, "wb");
    if (!fi) fatal_error("Failed to open %s for writing", path_int);
    size_t wi = fwrite(buf_i, sizeof(int), (size_t)n * 4, fi);
    (void)wi;
    fclose(fi);

    FILE* ff = fopen(path_float, "wb");
    if (!ff) fatal_error("Failed to open %s for writing", path_float);
    size_t wf = fwrite(buf_f, sizeof(float), (size_t)n * 4, ff);
    (void)wf;
    fclose(ff);

    /* Meta JSON */
    FILE* fm = fopen(path_meta, "w");
    if (!fm) fatal_error("Failed to open %s for writing", path_meta);
    fprintf(fm, "{\n");
    fprintf(fm, "  \"count\": %d,\n", n);
    fprintf(fm, "  \"int_file\": \"extrema_int.i32\",\n");
    fprintf(fm, "  \"float_file\": \"extrema_float.f32\",\n");
    fprintf(fm, "  \"int_order\": [\"o\", \"s\", \"i\", \"j\"],\n");
    fprintf(fm, "  \"float_order\": [\"y\", \"x\", \"sigma\", \"val\"]\n");
    fprintf(fm, "}\n");
    fclose(fm);

    xfree(buf_i);
    xfree(buf_f);
}

static void dump_refined_extrema_to_dir(const struct sift_keypoints* keys, const char* out_dir)
{
    ensure_dir_exists(out_dir);

    char path_int[FILENAME_MAX];
    char path_float[FILENAME_MAX];
    char path_meta[FILENAME_MAX];
    snprintf(path_int, sizeof(path_int), "%s/%s", out_dir, "extrema_refined_int.i32");
    snprintf(path_float, sizeof(path_float), "%s/%s", out_dir, "extrema_refined_float.f32");
    snprintf(path_meta, sizeof(path_meta), "%s/%s", out_dir, "extrema_refined_meta.json");

    int n = keys->size;
    int (*buf_i)[4] = (int (*)[4])xmalloc((size_t)n * 4 * sizeof(int));
    float (*buf_f)[4] = (float (*)[4])xmalloc((size_t)n * 4 * sizeof(float));

    for (int k = 0; k < n; k++) {
        const struct keypoint* key = keys->list[k];
        buf_i[k][0] = key->o;
        buf_i[k][1] = key->s;
        buf_i[k][2] = key->i;
        buf_i[k][3] = key->j;
        buf_f[k][0] = key->x; /* y_world */
        buf_f[k][1] = key->y; /* x_world */
        buf_f[k][2] = key->sigma;
        buf_f[k][3] = key->val; /* D_hat */
    }

    FILE* fi = fopen(path_int, "wb");
    if (!fi) fatal_error("Failed to open %s for writing", path_int);
    fwrite(buf_i, sizeof(int), (size_t)n * 4, fi);
    fclose(fi);

    FILE* ff = fopen(path_float, "wb");
    if (!ff) fatal_error("Failed to open %s for writing", path_float);
    fwrite(buf_f, sizeof(float), (size_t)n * 4, ff);
    fclose(ff);

    FILE* fm = fopen(path_meta, "w");
    if (!fm) fatal_error("Failed to open %s for writing", path_meta);
    fprintf(fm, "{\n");
    fprintf(fm, "  \"count\": %d,\n", n);
    fprintf(fm, "  \"int_file\": \"extrema_refined_int.i32\",\n");
    fprintf(fm, "  \"float_file\": \"extrema_refined_float.f32\",\n");
    fprintf(fm, "  \"int_order\": [\"o\", \"s\", \"i\", \"j\"],\n");
    fprintf(fm, "  \"float_order\": [\"y\", \"x\", \"sigma\", \"val\"]\n");
    fprintf(fm, "}\n");
    fclose(fm);

    xfree(buf_i);
    xfree(buf_f);
}

static void dump_oriented_keypoints_to_dir(const struct sift_keypoints* keys, const char* out_dir)
{
    ensure_dir_exists(out_dir);

    char path_int[FILENAME_MAX];
    char path_float[FILENAME_MAX];
    char path_desc[FILENAME_MAX];
    char path_meta[FILENAME_MAX];
    snprintf(path_int, sizeof(path_int), "%s/%s", out_dir, "keys_int.i32");
    snprintf(path_float, sizeof(path_float), "%s/%s", out_dir, "keys_float.f32");
    snprintf(path_desc, sizeof(path_desc), "%s/%s", out_dir, "keys_desc.u8");
    snprintf(path_meta, sizeof(path_meta), "%s/%s", out_dir, "keys_meta.json");

    int n = keys->size;
    int (*buf_i)[4] = (int (*)[4])xmalloc((size_t)n * 4 * sizeof(int));
    float (*buf_f)[4] = (float (*)[4])xmalloc((size_t)n * 4 * sizeof(float));
    int nd = 0;
    if (n > 0) {
        const struct keypoint* k0 = keys->list[0];
        nd = k0->n_hist * k0->n_hist * k0->n_ori;
        if (nd <= 0) nd = 128; /* fallback */
    }
    unsigned char* buf_d = (unsigned char*)xmalloc((size_t)(n > 0 ? n : 1) * (size_t)(nd > 0 ? nd : 128));

    for (int k = 0; k < n; k++) {
        const struct keypoint* key = keys->list[k];
        buf_i[k][0] = key->o;
        buf_i[k][1] = key->s;
        buf_i[k][2] = key->i;
        buf_i[k][3] = key->j;
        buf_f[k][0] = key->x;     /* y_world */
        buf_f[k][1] = key->y;     /* x_world */
        buf_f[k][2] = key->sigma; /* sigma */
        buf_f[k][3] = key->theta; /* orientation */
        if (nd == 0) {
            nd = key->n_hist * key->n_hist * key->n_ori;
            if (nd <= 0) nd = 128;
        }
        /* Write descriptor in u-major order (u, v, o) */
        int NH = key->n_hist;
        int NO = key->n_ori;
        for (int u = 0; u < NH; ++u) {
            for (int v = 0; v < NH; ++v) {
                for (int o = 0; o < NO; ++o) {
                    int src = (v * NH + u) * NO + o; /* in-memory v-major */
                    int dst = (u * NH + v) * NO + o; /* serialized u-major */
                    float vv = key->descr ? key->descr[src] : 0.0f;
                    if (vv < 0.0f) vv = 0.0f;
                    if (vv > 255.0f) vv = 255.0f;
                    buf_d[(size_t)k * (size_t)nd + (size_t)dst] = (unsigned char)(vv + 0.5f);
                }
            }
        }
    }

    FILE* fi = fopen(path_int, "wb");
    if (!fi) fatal_error("Failed to open %s for writing", path_int);
    fwrite(buf_i, sizeof(int), (size_t)n * 4, fi);
    fclose(fi);

    FILE* ff = fopen(path_float, "wb");
    if (!ff) fatal_error("Failed to open %s for writing", path_float);
    fwrite(buf_f, sizeof(float), (size_t)n * 4, ff);
    fclose(ff);

    FILE* fd = fopen(path_desc, "wb");
    if (!fd) fatal_error("Failed to open %s for writing", path_desc);
    if (n > 0 && nd > 0) {
        fwrite(buf_d, sizeof(unsigned char), (size_t)n * (size_t)nd, fd);
    }
    fclose(fd);

    FILE* fm = fopen(path_meta, "w");
    if (!fm) fatal_error("Failed to open %s for writing", path_meta);
    fprintf(fm, "{\n");
    fprintf(fm, "  \"count\": %d,\n", n);
    fprintf(fm, "  \"int_file\": \"keys_int.i32\",\n");
    fprintf(fm, "  \"float_file\": \"keys_float.f32\",\n");
    fprintf(fm, "  \"desc_file\": \"keys_desc.u8\",\n");
    fprintf(fm, "  \"desc_len\": %d,\n", nd);
    fprintf(fm, "  \"int_order\": [\"o\", \"s\", \"i\", \"j\"],\n");
    fprintf(fm, "  \"float_order\": [\"y\", \"x\", \"sigma\", \"theta\"]\n");
    fprintf(fm, "}\n");
    fclose(fm);

    xfree(buf_i);
    xfree(buf_f);
    xfree(buf_d);
}

