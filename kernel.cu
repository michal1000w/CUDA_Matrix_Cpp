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
    CMatrix<float> mat, mat2;
    /*
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

    cout << endl;
    mat += 2;
    mat.print();

    mat -= 2; 
    mat.print();

    Matrix<float> m3("[2,2,2][3,3,3]");
    mat2 = m3;
    mat2.print();

    mat2 = mat + m3;
    mat2.print();
    mat2 = mat2 - m3;
    mat2.print();
    mat2 = mat * m3;
    mat2.print();

    cout << endl << endl;
    */
    mat.add("[1,2,3]");
    mat2.add("[1,0][2,2][3,0]");

    
    mat = mat.dot(mat2) * 5;
    mat.print();
    cout << endl;
    mat.T().print();
    mat.sigmoid().print();
    

    mat.square().print();


    return 0;
}