//#include "CUDA_Class.cuh"
#include "CudaMatrix.cuh"



//////////////////////////funkcje pomocnicze////////////////
void addData(float* a, unsigned int size, bool zeros = false) {
    for (int i = 0; i < size; i++)
        if (zeros)
            a[i] = 0.0;
        else
            a[i] = i;
}

void printArray(float* a, unsigned int size) {
    cout << " [ ";
    for (int i = 0; i < size; i++)
        cout << a[i] << " ";
    cout << " ]" << endl;
}

int main() {
    /*
    unsigned int size = 5; //5
    float* a = new float[size];
    float* b = new float[size];

    CUDA_Class<float> cu;

    //add data
    addData(a, size);
    addData(b, size);

    //print data
    printArray(a, size);
    cout << " + " << endl;
    printArray(b, size);
    cout << " = " << endl;

    //launch kernel function
    cu.add(a, b, size);


    //print result
    printArray(a, size);

    //subtract
    cu.subtract(a, b, size);
    //cu.subtract(a, b, size);
    printArray(a, size);

    //mulitply by elements
    cu.multiply(a, b, size);
    printArray(a, size);

    //multiply by constant
    cu.cmultiply(a, 2, size);
    printArray(a, size);

    //divide by constant
    cu.cdivide(a, 2, size);
    cu.cdivide(a, 2, size);
    printArray(a, size);

    //add constant
    cu.cadd(a, 1, size);
    printArray(a, size);

    //subtract constant
    cu.csubtract(a, 1, size);
    printArray(a, size);
    */

    /////MATRIX CLASS
    Matrix<float> mat, mat2;
    mat.add("[1,2,3][4,5,6]");
    mat2.add("[1,2,3][4,5,6]");
    mat.print();

    mat += mat;
    mat.print();

    mat -= mat2;
    mat.print();

    mat *= mat2;
    mat.print();

    mat *= 2;
    mat.print();

    mat /= 2; 
    mat.print();


    return 0;
}