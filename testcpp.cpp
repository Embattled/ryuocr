#include <Python.h>

int main(int argc, char const *argv[])
{
    Py_Initialize();
    PyRun_SimpleString("print('hello world')\n");
    Py_Finalize();
    return 0;
}
