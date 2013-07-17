#include <Python.h>
#include <numpy/arrayobject.h>
#include "george.h"

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
*george_predict (PyObject *self, PyObject *args)
{
    double amp, var;
    PyObject *x_obj, *y_obj, *yerr_obj, *xtest_obj;

    // Parse the input arguments.
    if (!PyArg_ParseTuple(args, "OOOddO", &x_obj, &y_obj, &yerr_obj, &amp,
                          &var, &xtest_obj))
        return NULL;

    // Decode the numpy arrays.
    PyArrayObject *x_array = PARSE_ARRAY(x_obj),
                  *y_array = PARSE_ARRAY(y_obj),
                  *yerr_array = PARSE_ARRAY(yerr_obj),
                  *xtest_array = PARSE_ARRAY(xtest_obj);
    if (x_array == NULL || y_array == NULL || yerr_array == NULL ||
        xtest_array == NULL)
        goto fail;

    // Get the dimensions.
    int nsamples = (int) PyArray_DIM(x_array, 0),
        ntest = (int) PyArray_DIM(xtest_array, 0);
    if ((int)PyArray_DIM(y_array, 0) != nsamples ||
        (int)PyArray_DIM(yerr_array, 0) != nsamples)
    {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        goto fail;
    }

    // Allocate the output array.
    npy_intp dim[1] = {ntest}, dim2[2] = {ntest, ntest};
    PyArrayObject *mean_array = (PyArrayObject*)PyArray_SimpleNew(1, dim,
                                                                  NPY_DOUBLE),
                  *cov_array = (PyArrayObject*)PyArray_SimpleNew(2, dim2,
                                                                 NPY_DOUBLE);
    if (mean_array == NULL || cov_array == NULL) {
        Py_XDECREF(mean_array);
        Py_XDECREF(cov_array);
        goto fail;
    }

    // Compute the light curve.
    double *x = PyArray_DATA(x_array),
           *y = PyArray_DATA(y_array),
           *yerr = PyArray_DATA(yerr_array),
           *xtest = PyArray_DATA(xtest_array),
           *mean = PyArray_DATA(mean_array),
           *cov = PyArray_DATA(cov_array);
    int info = gp_predict (nsamples, x, y, yerr, amp, var, ntest, xtest,
                           mean, cov);

    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(yerr_array);
    Py_DECREF(xtest_array);

    if (info != 0) {
        PyErr_SetString(PyExc_RuntimeError, "GP predict failed.");
        Py_DECREF(mean_array);
        Py_DECREF(cov_array);
        return NULL;
    }

    return Py_BuildValue("OO", mean_array, cov_array);

fail:

    Py_XDECREF(x_array);
    Py_XDECREF(y_array);
    Py_XDECREF(yerr_array);
    Py_XDECREF(xtest_array);
    return NULL;

}

static PyObject
*george_lnlikelihood (PyObject *self, PyObject *args)
{
    double amp, var;
    PyObject *x_obj, *y_obj, *yerr_obj;

    // Parse the input arguments.
    if (!PyArg_ParseTuple(args, "OOOdd", &x_obj, &y_obj, &yerr_obj, &amp, &var))
        return NULL;

    // Decode the numpy arrays.
    PyArrayObject *x_array = PARSE_ARRAY(x_obj),
                  *y_array = PARSE_ARRAY(y_obj),
                  *yerr_array = PARSE_ARRAY(yerr_obj);
    if (x_array == NULL || y_array == NULL || yerr_array == NULL)
        goto fail;

    // Get the dimensions.
    int nsamples = (int) PyArray_DIM(x_array, 0);
    if ((int)PyArray_DIM(y_array, 0) != nsamples ||
        (int)PyArray_DIM(yerr_array, 0) != nsamples) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        goto fail;
    }

    // Compute the light curve.
    double *x = PyArray_DATA(x_array),
           *y = PyArray_DATA(y_array),
           *yerr = PyArray_DATA(yerr_array);
    double lnlike = gp_lnlikelihood (nsamples, x, y, yerr, amp, var);

    // Clean up.
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(yerr_array);

    return Py_BuildValue("d", lnlike);

fail:

    Py_XDECREF(x_array);
    Py_XDECREF(y_array);
    Py_XDECREF(yerr_array);
    return NULL;

}

static PyObject
*george_gradlnlikelihood (PyObject *self, PyObject *args)
{
    double amp, var;
    PyObject *x_obj, *y_obj, *yerr_obj;

    // Parse the input arguments.
    if (!PyArg_ParseTuple(args, "OOOdd", &x_obj, &y_obj, &yerr_obj, &amp, &var))
        return NULL;

    // Decode the numpy arrays.
    PyArrayObject *x_array = PARSE_ARRAY(x_obj),
                  *y_array = PARSE_ARRAY(y_obj),
                  *yerr_array = PARSE_ARRAY(yerr_obj);
    if (x_array == NULL || y_array == NULL || yerr_array == NULL)
        goto fail;

    // Get the dimensions.
    int nsamples = (int) PyArray_DIM(x_array, 0);
    if ((int)PyArray_DIM(y_array, 0) != nsamples ||
        (int)PyArray_DIM(yerr_array, 0) != nsamples) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        goto fail;
    }

    // Compute the light curve.
    double *x = PyArray_DATA(x_array),
           *y = PyArray_DATA(y_array),
           *yerr = PyArray_DATA(yerr_array);
    double lnlike, *gradlnlike = malloc(2 * sizeof(double));
    int info = gp_gradlnlikelihood (nsamples, x, y, yerr, amp, var, &lnlike,
                                    &gradlnlike);

    // Clean up.
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(yerr_array);

    if (info != 0) {
        free(gradlnlike);
        return NULL;
    }

    // Build the output.
    npy_intp dim[1] = {2};
    PyObject *grad = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, gradlnlike);
    if (grad == NULL) {
        free(gradlnlike);
        return NULL;
    }

    return Py_BuildValue("dO", lnlike, grad);

fail:

    Py_XDECREF(x_array);
    Py_XDECREF(y_array);
    Py_XDECREF(yerr_array);
    return NULL;

}

static PyMethodDef george_methods[] = {
    {"predict",
     (PyCFunction) george_predict,
     METH_VARARGS,
     "Predict a GP."},
    {"lnlikelihood",
     (PyCFunction) george_lnlikelihood,
     METH_VARARGS,
     "Compute the ln-likelihood of a GP."},
    {"gradlnlikelihood",
     (PyCFunction) george_gradlnlikelihood,
     METH_VARARGS,
     "Compute the ln-likelihood and gradients of a GP."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int george_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int george_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_george",
    NULL,
    sizeof(struct module_state),
    george_methods,
    NULL,
    george_traverse,
    george_clear,
    NULL
};

#define INITERROR return NULL

PyObject *PyInit__george(void)

#else
#define INITERROR return

void init_george(void)

#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_george", george_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_george.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
