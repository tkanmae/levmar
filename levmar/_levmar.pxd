# -*- coding: utf-8 -*-

cdef extern from "levmar.h":
    enum:
        LM_OPTS_SZ
        LM_INFO_SZ
        LM_ERROR

    double LM_INIT_MU
    double LM_STOP_THRESH
    double LM_DIFF_DELTA

    # Work arrays size for `levmar_der` and `levmar_dif` functions.
    # Should be multiplied by sizeof(double) or sizeof(float) to be
    # converted to bytes
    int LM_DER_WORKSZ(int npar, int nmeas)
    int LM_DIF_WORKSZ(int npar, int nmeas)
    # Work arrays size for ?levmar_bc_der and ?levmar_bc_dif functions.
    # Should be multiplied by sizeof(double) or sizeof(float) to be
    # converted to bytes
    int LM_BC_DER_WORKSZ(int npar, int nmeas)
    int LM_BC_DIF_WORKSZ(int npar, int nmeas)
    # Work arrays size for ?levmar_lec_der and ?levmar_lec_dif functions.
    # Should be multiplied by sizeof(double) or sizeof(float) to be
    # converted to bytes
    int LM_LEC_DER_WORKSZ(int npar, int nmeas, int nconstr)
    int LM_LEC_DIF_WORKSZ(int npar, int nmeas, int nconstr)
    # Work arrays size for ?levmar_blec_der and ?levmar_blec_dif functions.
    # Should be multiplied by sizeof(double) or sizeof(float) to be
    # converted to bytes
    int LM_BLEC_DER_WORKSZ(int npar, int nmeas, int nconstr)
    int LM_BLEC_DIF_WORKSZ(int npar, int nmeas, int nconstr)
    int LM_BLEIC_DER_WORKSZ(int npar, int nmeas, int nconstr1, int nconstr2)
    int LM_BLEIC_DIF_WORKSZ(int npar, int nmeas, int nconstr1, int nconstr2)

    # Unconstrained minimization
    int dlevmar_der(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        void (*jacf)(double *p, double *j, int m, int n, void *adata),
        double *p, double *x, int m, int n, int itmax, double *opts,
        double *info, double *work, double *covar, void *adata)

    int dlevmar_dif(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        double *p, double *x, int m, int n, int itmax, double *opts,
        double *info, double *work, double *covar, void *adata)

    # Box-constrained minimization
    int dlevmar_bc_der(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        void (*jacf)(double *p, double *j, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *lb, double *ub, double *dscl,
        int itmax, double *opts, double *info, double *work, double *covar, void *adata)

    int dlevmar_bc_dif(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *lb, double *ub, double *dscl,
        int itmax, double *opts, double *info, double *work, double *covar, void *adata)

    # Linear equation constrained minimization
    int dlevmar_lec_der(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        void (*jacf)(double *p, double *j, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *A, double *b, int k,
        int itmax, double *opts, double *info, double *work, double *covar, void *adata)

    int dlevmar_lec_dif(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *A, double *b, int k,
        int itmax, double *opts, double *info, double *work, double *covar, void *adata)

    # Box & linear equation constrained minimization
    int dlevmar_blec_der(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        void (*jacf)(double *p, double *j, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *lb, double *ub, double *A, double *b, int k, double *wghts,
        int itmax, double *opts, double *info, double *work, double *covar, void *adata)

    int dlevmar_blec_dif(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *lb, double *ub, double *A, double *b, int k, double *wghts,
        int itmax, double *opts, double *info, double *work, double *covar, void *adata)

    # Box, linear equations & inequalities constrained minimization
    int dlevmar_bleic_der(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        void (*jacf)(double *p, double *j, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *lb, double *ub,
        double *A, double *b, int k1, double *C, double *d, int k2,
        int itmax, double *opts, double *info, double *work, double *covar, void *adata)

    int dlevmar_bleic_dif(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *lb, double *ub,
        double *A, double *b, int k1, double *C, double *d, int k2,
        int itmax, double *opts, double *info, double *work, double *covar, void *adata)

    # Box & linear inequality constraints
    int dlevmar_blic_der(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        void (*jacf)(double *p, double *j, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *lb, double *ub, double *C, double *d, int k2,
        int itmax, double opts[4], double info[LM_INFO_SZ], double *work, double *covar, void *adata)

    int dlevmar_blic_dif(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *lb, double *ub, double *C, double *d, int k2,
        int itmax, double opts[5], double info[LM_INFO_SZ], double *work, double *covar, void *adata)

    # Linear equation & inequality constraints
    int dlevmar_leic_der(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        void (*jacf)(double *p, double *j, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *A, double *b, int k1, double *C, double *d, int k2,
        int itmax, double opts[4], double info[LM_INFO_SZ], double *work, double *covar, void *adata)

    int dlevmar_leic_dif(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *A, double *b, int k1, double *C, double *d, int k2,
        int itmax, double opts[5], double info[LM_INFO_SZ], double *work, double *covar, void *adata)

    # Linear inequality constraints
    int dlevmar_lic_der(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        void (*jacf)(double *p, double *j, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *C, double *d, int k2,
        int itmax, double opts[4], double info[LM_INFO_SZ], double *work, double *covar, void *adata)

    int dlevmar_lic_dif(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        double *p, double *x, int m, int n, double *C, double *d, int k2,
        int itmax, double opts[5], double info[LM_INFO_SZ], double *work, double *covar, void *adata)

    # Jacobian verification, double  precision
    void dlevmar_chkjac(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        void (*jacf)(double *p, double *j, int m, int n, void *adata),
        double *p, int m, int n, void *adata, double *err)

    # Standard deviation, coefficient of determination (R2) & Pearson's
    # Correlation coefficient for best-fit parameters
    double dlevmar_stddev( double *covar, int m, int i)
    double dlevmar_corcoef(double *covar, int m, int i, int j)
    double dlevmar_R2(
        void (*func)(double *p, double *hx, int m, int n, void *adata),
        double *p, double *x, int m, int n, void *adata)
