#include "CudaMatrix.cuh"
#include <chrono>



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

void print_time(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point stop) {
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    cout << "[Done] in : ";

    double seconds = double((int)duration % 60000) / 1000.0;
    int minutes = int(duration / 1000) / 60;

    if (minutes > 0)
        cout << "[ " << minutes << " min " << seconds << " s ]" << endl;
    else
        cout << "[ " << seconds << " s ]" << endl;
}

int main() {
    auto start = std::chrono::steady_clock::now();

    CuMatrix<float> mat, mat2;
    mat.add("[1,2,3][4,5,6]");
    mat2.add("[2,2,2][3,3,3]");

    mat += mat2;
    mat.print();

    mat -= mat2;
    mat.print();

    mat += 2;
    mat.print();

    mat -= 2; 
    mat.print();

    cout << endl << endl;
    mat *= mat2;
    mat.print();

    mat *= 2;
    mat.print();

    mat /= 2;
    mat.print();

    cout << endl << endl;
    mat.add("[1,2,3][4,5,6]");
    mat2.add("[2,2,2][3,3,3]");

    mat = mat + mat;
    mat.print();
    mat = mat - mat;
    mat.print();

    cout << endl << endl;
    mat.add("[1,2,3][4,5,6]");
    mat2.add("[2,2,2][3,3,3]");

    mat = mat + 2;
    mat.print();
    mat = mat - 2;
    mat.print();
    mat = mat * 2;
    mat.print();
    mat = mat / 2;
    mat.print();
    mat = mat * mat2;
    mat.print();
    
    cout << endl << endl;
    mat.sigmoid().print();
    mat.sigmoid_derivative().print();
    mat.square().print();

    cout << endl << endl;
    mat.add("[1,2,3][4,5,6]");
    mat2.add("[1,0][0,2][3,0]");

    mat.dot(mat2).print();
    mat.T().print();


    ///sprawdzanie czasu
    auto stop = std::chrono::steady_clock::now();
    print_time(start, stop);
    return 0;
}