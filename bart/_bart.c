#include <Python.h>
#include <numpy/arrayobject.h>
#include "bart.h"

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#define PARSE_ARRAY(o) (PyArrayObject*) PyArray_FROM_OTF(o, NPY_DOUBLE, \
        NPY_IN_ARRAY)

static PyObject
*bart_lightcurve(PyObject *self, PyObject *args)
{
    int K;
    double texp, fstar, mstar, rstar, iobs;
    PyObject *t_obj, *mass_obj, *r_obj, *a_obj, *t0_obj, *e_obj, *pomega_obj,
             *ix_obj, *iy_obj, *ldp_r_obj, *ldp_I_obj;

    // Parse the input arguments.
    if (!PyArg_ParseTuple(args, "OdidddOOOOOOOOOO", &t_obj, &texp, &K,
                          &fstar, &mstar, &rstar, &iobs,
                          &mass_obj, &r_obj, &a_obj, &t0_obj, &e_obj,
                          &pomega_obj, &ix_obj, &iy_obj,
                          &ldp_r_obj, &ldp_I_obj))
        return NULL;

    // Decode the numpy arrays.
    PyArrayObject *t_array = PARSE_ARRAY(t_obj),
                  *mass_array = PARSE_ARRAY(mass_obj),
                  *r_array = PARSE_ARRAY(r_obj),
                  *a_array = PARSE_ARRAY(a_obj),
                  *t0_array = PARSE_ARRAY(t0_obj),
                  *e_array = PARSE_ARRAY(e_obj),
                  *pomega_array = PARSE_ARRAY(pomega_obj),
                  *ix_array = PARSE_ARRAY(ix_obj),
                  *iy_array = PARSE_ARRAY(iy_obj),
                  *ldp_r_array = PARSE_ARRAY(ldp_r_obj),
                  *ldp_I_array = PARSE_ARRAY(ldp_I_obj);
    if (t_array == NULL || mass_array == NULL || r_array == NULL ||
        a_array == NULL || t0_array == NULL || e_array == NULL ||
        pomega_array == NULL || ix_array == NULL || iy_array == NULL ||
        ldp_r_array == NULL || ldp_I_array == NULL)
        goto fail;

    // Check dimensions.
    int n = (int) PyArray_DIM(t_array, 0),
        np = (int) PyArray_DIM(mass_array, 0),
        nld = (int) PyArray_DIM(ldp_r_array, 0);
    if ((int)PyArray_DIM(r_array, 0) != np ||
        (int)PyArray_DIM(a_array, 0) != np ||
        (int)PyArray_DIM(t0_array, 0) != np ||
        (int)PyArray_DIM(e_array, 0) != np ||
        (int)PyArray_DIM(pomega_array, 0) != np ||
        (int)PyArray_DIM(ix_array, 0) != np ||
        (int)PyArray_DIM(iy_array, 0) != np ||
        (int)PyArray_DIM(ldp_I_array, 0) != nld)
    {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        goto fail;
    }

    // Allocate the flux array.
    npy_intp dim[1] = {n};
    PyArrayObject *flux_array = (PyArrayObject*)PyArray_SimpleNew(1, dim,
                                                                  NPY_DOUBLE);

    // Compute the light curve.
    int info = lightcurve (n, PyArray_DATA(t_array), PyArray_DATA(flux_array),
                           K, texp,
                           fstar, mstar, rstar, iobs,
                           np, PyArray_DATA(mass_array),
                           PyArray_DATA(r_array),
                           PyArray_DATA(a_array),
                           PyArray_DATA(t0_array),
                           PyArray_DATA(e_array),
                           PyArray_DATA(pomega_array),
                           PyArray_DATA(ix_array),
                           PyArray_DATA(iy_array),
                           nld, PyArray_DATA(ldp_r_array),
                           PyArray_DATA(ldp_I_array));

    Py_DECREF(t_array);
    Py_DECREF(mass_array);
    Py_DECREF(r_array);
    Py_DECREF(a_array);
    Py_DECREF(t0_array);
    Py_DECREF(e_array);
    Py_DECREF(pomega_array);
    Py_DECREF(ix_array);
    Py_DECREF(iy_array);
    Py_DECREF(ldp_r_array);
    Py_DECREF(ldp_I_array);

    return Py_BuildValue("O", &flux_array);

fail:

    Py_XDECREF(t_array);
    Py_XDECREF(mass_array);
    Py_XDECREF(r_array);
    Py_XDECREF(a_array);
    Py_XDECREF(t0_array);
    Py_XDECREF(e_array);
    Py_XDECREF(pomega_array);
    Py_XDECREF(ix_array);
    Py_XDECREF(iy_array);
    Py_XDECREF(ldp_r_array);
    Py_XDECREF(ldp_I_array);
    return NULL;

}

static PyMethodDef bart_methods[] = {
    {"lightcurve",
     (PyCFunction)bart_lightcurve,
     METH_VARARGS,
     "Generate a light curve for a set of planet parameters."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int bart_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int bart_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_bart",
    NULL,
    sizeof(struct module_state),
    bart_methods,
    NULL,
    bart_traverse,
    bart_clear,
    NULL
};

#define INITERROR return NULL

PyObject *PyInit__bart(void)

#else
#define INITERROR return

void init_bart(void)

#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_bart", bart_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_bart.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
