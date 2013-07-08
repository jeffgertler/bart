#include <Python.h>
#include <numpy/arrayobject.h>
#include "turnstile.h"

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#define PARSE_ARRAY(o) (PyArrayObject*) PyArray_FROM_OTF(o, NPY_DOUBLE, NPY_IN_ARRAY)

static PyObject
*turnstile_period_search (PyObject *self, PyObject *args)
{
    int i, j;

    int nperiods, ndepths;
    double min_period, max_period, min_depth, max_depth,  amp, var;
    PyObject *datasets;

    // Parse the input arguments.
    if (!PyArg_ParseTuple(args, "Oddiddidd", &datasets, &min_period,
                          &max_period, &nperiods, &min_depth, &max_depth,
                          &ndepths, &amp, &var))
        return NULL;

    // Allocate memory for outputs.
    npy_intp dim[1] = {nperiods};
    PyArrayObject *periodsarray = (PyArrayObject*)PyArray_SimpleNew(1, dim,
                                                                    NPY_DOUBLE),
                  *depthsarray = (PyArrayObject*)PyArray_SimpleNew(1, dim,
                                                                   NPY_DOUBLE);
    if (periodsarray == NULL || depthsarray == NULL) {
        Py_XDECREF(periodsarray);
        Py_XDECREF(depthsarray);
        return NULL;
    }

    // Parse and extract the necessary information from the datasets.
    if (!PyList_Check(datasets)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list.");
        Py_DECREF(periodsarray);
        Py_DECREF(depthsarray);
        return NULL;
    }
    int max_sets, nsets = (int)PyList_Size(datasets),
        *ndata = malloc(nsets * sizeof(int));
    double **time = malloc(nsets * sizeof(double*)),
           **flux = malloc(nsets * sizeof(double*)),
           **ferr = malloc(nsets * sizeof(double*));
    PyObject *timeobj = NULL, *fluxobj = NULL, *ferrobj = NULL;
    PyArrayObject *timearray = NULL, *fluxarray = NULL, *ferrarray = NULL;

    for (max_sets = 0; max_sets < nsets; ++max_sets) {
        // Get the dataset.
        PyObject *ds = PyList_GetItem(datasets, max_sets);
        if (ds == NULL) goto fail;

        // Access the attributes that we need.
        timeobj = PyObject_GetAttrString(ds, "time");
        fluxobj = PyObject_GetAttrString(ds, "flux");
        ferrobj = PyObject_GetAttrString(ds, "ferr");

        // Clean up properly if anything went wrong.
        if (timeobj == NULL || fluxobj == NULL || ferrobj == NULL)
            goto ds_fail;

        // Parse the objects as numpy arrays.
        timearray = PARSE_ARRAY(timeobj),
        fluxarray = PARSE_ARRAY(fluxobj),
        ferrarray = PARSE_ARRAY(ferrobj);

        // Clean up properly if anything went wrong.
        if (timearray == NULL || fluxarray == NULL || ferrarray == NULL)
            goto ds_fail;

        // Figure out the size of the dataset and allocate the needed memory.
        ndata[max_sets] = (int)PyArray_DIM(timearray, 0);
        if (PyArray_DIM(fluxarray, 0) != ndata[max_sets] ||
            PyArray_DIM(ferrarray, 0) != ndata[max_sets]) {
            PyErr_SetString(PyExc_ValueError, "Dimension mismatch.");
            goto ds_fail;
        }
        time[max_sets] = malloc(ndata[max_sets] * sizeof(double));
        flux[max_sets] = malloc(ndata[max_sets] * sizeof(double));
        ferr[max_sets] = malloc(ndata[max_sets] * sizeof(double));

        // Copy the data over.
        double *t =  PyArray_DATA(timearray),
               *f =  PyArray_DATA(fluxarray),
               *fe =  PyArray_DATA(ferrarray);
        for (i = 0; i < ndata[max_sets]; ++i) {
            time[max_sets][i] = t[i];
            flux[max_sets][i] = f[i];
            ferr[max_sets][i] = fe[i];
        }

        // Reference counting.
        Py_DECREF(timeobj);
        Py_DECREF(fluxobj);
        Py_DECREF(ferrobj);
        Py_DECREF(timearray);
        Py_DECREF(fluxarray);
        Py_DECREF(ferrarray);
    }

    double *periods = PyArray_DATA(periodsarray),
           *depths = PyArray_DATA(depthsarray);
    turnstile (nsets, ndata, time, flux, ferr, amp, var,
               min_period, max_period, nperiods,
               min_depth, max_depth, ndepths, periods, depths);

    return Py_BuildValue("OO", periodsarray, depthsarray);

ds_fail:

    Py_XDECREF(timeobj);
    Py_XDECREF(fluxobj);
    Py_XDECREF(ferrobj);
    Py_XDECREF(timearray);
    Py_XDECREF(fluxarray);
    Py_XDECREF(ferrarray);

fail:

    Py_DECREF(periodsarray);
    Py_DECREF(depthsarray);
    for (j = 0; j < max_sets; ++j) {
        free(time[j]);
        free(flux[j]);
        free(ferr[j]);
    }
    free(time);
    free(flux);
    free(ferr);
    return NULL;
}

static PyMethodDef turnstile_methods[] = {
    {"period_search",
     (PyCFunction) turnstile_period_search,
     METH_VARARGS,
     "Search for periods."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int turnstile_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int turnstile_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_turnstile",
    NULL,
    sizeof(struct module_state),
    turnstile_methods,
    NULL,
    turnstile_traverse,
    turnstile_clear,
    NULL
};

#define INITERROR return NULL

PyObject *PyInit__turnstile(void)

#else
#define INITERROR return

void init_turnstile(void)

#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_turnstile", turnstile_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_turnstile.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
