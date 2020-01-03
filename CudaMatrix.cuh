#ifndef __CUDA_MATRIX__
#define __CUDA_MATRIX__


#include "CUDA_Class.cuh";

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

typedef long long yeet;
using std::vector;
using std::string;

template <typename Y>
void deletep(Y&) {}
template <typename Y>
void deletep(Y*& ptr) {
    delete ptr;
    ptr = nullptr;
}
template <typename Y>
void deletea(Y*& ptr) {
    if (ptr != nullptr) {
        delete[] ptr;
        ptr = nullptr;
    }
}
template <typename Y>
class Matrix;

//////////////////////////////////////////////////KERNELS//////////////////////////////////////////////////////

template <typename Y>
__global__ 
void dotKernel(Y* a, Y* b, Y* c, yeet m, yeet n, yeet k) {
    yeet row = blockIdx.y * blockDim.y + threadIdx.y;
    yeet col = blockIdx.x * blockDim.x + threadIdx.x;
    Y sum = 0;
    if (col < k && row < m) {
        for (yeet i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

template <typename Y>
__global__
void transposeKernel(Y* mat_in, Y* mat_out, yeet rows, yeet cols) {
    yeet idx = blockIdx.x * blockDim.x + threadIdx.x;
    yeet idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        yeet pos = idy * cols + idx;
        yeet trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}


//////////////////////////////////////////////////CLASS////////////////////////////////////////////////////////
template<typename Y>
class Matrix {
public:
    typedef Y value_type;
    Matrix() : _cols(0), _rows(0), _data(new Y[0]) {};
    Matrix(yeet rows, yeet cols);
    //new
    Matrix(yeet, yeet, mutable Y*);
    Matrix(yeet, yeet, vector<Y>);
    Matrix(const std::string&);
    Matrix(const Matrix<Y>&);

    bool exists(yeet row, yeet col) const;
    Y& operator()(yeet row, yeet col);
    Y operator()(yeet row, yeet col) const;
    Matrix<Y> operator=(const Matrix<Y>&);
    virtual ~Matrix();

    //friend void dotKernel(const Matrix<Y>* A, const Matrix<Y>* B, Matrix<Y>* C);


    /////NOWE
    Matrix<Y> operator+=(const Matrix<Y>&);
    Matrix<Y> operator+=(const Y&);
    Matrix<Y> operator-=(const Matrix<Y>&);
    Matrix<Y> operator-=(const Y&);
    Matrix<Y> operator*=(const Matrix<Y>&);
    Matrix<Y> operator*=(const Y&);
    Matrix<Y> operator/=(const Y&);

    //Dodatkowe
    Matrix<Y> operator+(const Matrix<Y>&);
    Matrix<Y> operator-(const Matrix<Y>&);
    Matrix<Y> operator*(const Matrix<Y>&);
    Matrix<Y> operator*(const Y&);
    Matrix<Y> operator/(const Y&);

    Matrix<Y> dot(const Matrix<Y>&);
    Matrix<Y> T();

    Matrix<Y> sigmoid();
    Matrix<Y> sigmoid_derivative();

    double mean(); //for now works only on CPU //todo gpu compute
    Matrix<Y> square();


    //friend inline Matrix<Y>& sigmoid(const Matrix<Y>& m);
    //friend inline Matrix<Y>& sigmoid_derivative(const Matrix<Y>& m);

    Matrix<Y> print(short round = (short)5);
    void add(std::string);
    std::string getString(bool is_int = false);

    //////////

    yeet size() const { return _rows * _cols; }
    yeet rows() const { return _rows; }
    yeet cols() const { return _cols; }

    Y* getArray() const {
        Y* temp = new Y[size()];
        for (yeet i = 0; i < size(); i++)
            temp[i] = _data[i];
        return temp;
    }
    void print_size() { std::cout << "size : " << _rows << " x " << _cols << std::endl; }

    //setters
    void set_threads_per_block(unsigned int threads = 32) { this->THREADS_PER_BLOCK = threads; }
private:
    unsigned power(short inp) {
        unsigned int output = 1;
        for (short i = 0; i < inp; i++)
            output *= 10;
        return output;
    }
    void initMatrix() {
        this->_data = new Y[size()];
        //for (yeet i = 0; i < size(); i++) this->_data[i] = 0.0;
        Cuda.cset(this->_data, 0.0, size());
    }
    yeet _rows, _cols;
    mutable Y* _data;

    CUDA_Class<Y> Cuda;
    unsigned int THREADS_PER_BLOCK = 32;
public:
    void free();
};

//////////////////////////////////////////////////KONSTRUKTORY///////////////////////////////////////////////////////////////
template<typename Y>
Matrix<Y>::Matrix(yeet rows, yeet cols) : _rows(rows), _cols(cols) {
    this->initMatrix();
}

template <typename Y>
Matrix<Y>::Matrix(yeet rows, yeet cols, vector<Y> data) : _rows(rows), _cols(cols) {
    this->initMatrix();
    for (yeet i = 0; i < data.size(); i++)
        _data[i] = data[i];
}

template<typename Y>
Matrix<Y>::Matrix(yeet rows, yeet cols, mutable Y* _data) : _rows(rows), _cols(cols) {
    this->initMatrix();
    //for (yeet i = 0; i < size(); i++) this->_data[i] = _data[i];
    Cuda.set(this->_data, _data, size());
}
template <typename Y>
Matrix<Y>::Matrix(const Matrix<Y>& mat) {
    this->_cols = mat._cols;
    this->_rows = mat._rows;
    this->initMatrix();
    Cuda.set(this->_data, mat._data, this->size());
}
template<typename Y>
Matrix<Y>::Matrix(const std::string& macierz) {
    //Podzia³ na fragmenty
    yeet len = macierz.length();
    std::string fragment = "";
    vector <std::string> fragmenty;

    for (yeet i = 0; i < len; i++) {
        if (macierz[i] == '[') {
            fragment = "";
            do {
                i++;
                if (macierz[i] == ']') break;

                fragment += macierz[i];
            } while (i < len - 1);
            fragmenty.push_back(fragment);
        }
    }

    this->_rows = fragmenty.size();

    //Podzia³ fragmentów na pojedyncze elementy
    vector <double> wartosci;
    short elementy = 0;

    for (yeet i = 0; i < this->_rows; i++) {
        len = fragmenty[i].size(); //d³ugoœæ stringa

        for (yeet j = 0; j < len; j++) {
            fragment = "";
            while (fragmenty[i][j] != ',' && j < len) {
                fragment += fragmenty[i][j];
                j++;
            }
            string::size_type sz;
            wartosci.push_back(stod(fragment, &sz));
        }

        if (i == 0) elementy = wartosci.size();
    }

    this->_cols = elementy;

    //Inicjowanie nowej macierzy typu Matrix
    this->initMatrix();

    //Przenoszenie elementów z vector do macierzy typu Matrix
    yeet i = 0;
    for (yeet y = 0; y < this->_rows; y++) {
        for (yeet x = 0; x < this->_cols; x++) {
            this->_data[y * this->_cols + x] = wartosci[i];
            i++;
        }
    }

    //czyszczenie pamiêci
    fragmenty.clear();
    wartosci.clear();
}

//////////////////////////////////////////////////DESTRUKTOR///////////////////////////////////////////////////////////////
template<typename Y>
Matrix<Y>::~Matrix() {
    this->free(); //tutaj
}
//////////////////////////////////////////////////DODATKOWE///////////////////////////////////////////////////////////////
template<typename Y>
Matrix<Y> Matrix<Y>::print(short roundness) {
    int pomocnicza = 0;
    roundness = (roundness < 5 ? roundness : 4);
    for (yeet i = 0; i < this->_rows; i++) {
        std::cout << "[";
        for (yeet j = 0; j < this->_cols; j++) {
            if (roundness != 0) {
                pomocnicza = (float)this->_data[i * this->_cols + j] * power(roundness);
                std::cout << " " << ((float)pomocnicza / power(roundness));
            }
            else
                std::cout << " " << round(this->_data[i * this->_cols + j]);
        }
        std::cout << " ]" << std::endl;
    }
    return *this;
}

template <typename Y>
std::string Matrix<Y>::getString(bool is_int) {
    std::string data = "";
    Matrix<Y> temp(*this);
    for (int i = 0; i < this->_rows; i++) {
        data += "[";
        for (int j = 0; j < this->_cols; j++) {
            if (!(is_int)) data += to_string(temp(i, j));
            else data += to_string(int(temp(i, j)));

            if (j < this->_cols - 1) data += ",";
        }
        data += "]";
    }
    return data;
}

template<typename Y>
bool Matrix<Y>::exists(yeet row, yeet col) const {
    return (row < _rows && col < _cols);
}

template<typename Y>
void Matrix<Y>::free() {
    for (yeet i = 0, c = size(); i < c; ++i) {
        //will do nothing if Y isn't a pointer
        deletep(_data[i]);
    }
    deletea(_data);
}

template <typename Y>
void Matrix<Y>::add(std::string macierz) {
    //Podzia³ na fragmenty
    yeet len = macierz.length();
    std::string fragment = "";
    vector <std::string> fragmenty;

    for (yeet i = 0; i < len; i++) {
        if (macierz[i] == '[') {
            fragment = "";
            do {
                i++;
                if (macierz[i] == ']') break;

                fragment += macierz[i];
            } while (i < len - 1);
            fragmenty.push_back(fragment);
        }
    }

    this->_rows = fragmenty.size();

    //Podzia³ fragmentów na pojedyncze elementy
    vector <double> wartosci;
    short elementy = 0;

    for (yeet i = 0; i < this->_rows; i++) {
        len = fragmenty[i].size(); //d³ugoœæ stringa

        for (yeet j = 0; j < len; j++) {
            fragment = "";
            while (fragmenty[i][j] != ',' && j < len) {
                fragment += fragmenty[i][j];
                j++;
            }
            string::size_type sz;
            wartosci.push_back(stod(fragment, &sz));
        }

        if (i == 0) elementy = wartosci.size();
    }

    this->_cols = elementy;

    //czyszczenie starej macierzy
    this->free();

    //Inicjowanie nowej macierzy typu Matrix
    this->initMatrix();

    //Przenoszenie elementów z vector do macierzy typu Matrix
    yeet i = 0;
    for (yeet y = 0; y < this->_rows; y++) {
        for (yeet x = 0; x < this->_cols; x++) {
            this->_data[y * this->_cols + x] = wartosci[i];
            i++;
        }
    }

    //czyszczenie pamiêci
    fragmenty.clear();
    wartosci.clear();
}



///////////////////////////////////////////////////////////////OPERATORY//////////////////////////////////////////////////////////////
/////////////////Operatory//////////////////////////////////////
template <typename Y>
Y& Matrix<Y>::operator()(yeet row, yeet col) {
    return _data[_cols * row + col];
}

template <typename Y>
Y Matrix<Y>::operator()(yeet row, yeet col) const {
    return _data[_cols * row + col];
}

template<typename Y>
Matrix<Y> Matrix<Y>::operator=(const Matrix<Y>& rhs) {
    this->free();

    this->_rows = rhs._rows;
    this->_cols = rhs._cols;

    this->initMatrix();
    Cuda.set(this->_data, rhs._data, this->size());

    return *this;
}

template <typename Y>
Matrix<Y> Matrix<Y>::operator+=(const Matrix<Y>& rhs) {
    Cuda.add(this->_data, rhs._data, this->size());
    return *this;
}
template <typename Y>
Matrix<Y> Matrix<Y>::operator+=(const Y& rhs) {
    Cuda.cadd(this->_data, rhs, this->size());
    return *this;
}
template <typename Y>
Matrix<Y> Matrix<Y>::operator-=(const Matrix<Y>& rhs) {
    Cuda.subtract(this->_data, rhs._data, this->size());
    return *this;
}
template <typename Y>
Matrix<Y> Matrix<Y>::operator-=(const Y& rhs) {
    Cuda.csubtract(this->_data, rhs, this->size());
    return *this;
}
template <typename Y>
Matrix<Y> Matrix<Y>::operator*=(const Matrix<Y>& rhs) {
    Cuda.multiply(this->_data, rhs._data, this->size());
    return *this;
}
template <typename Y>
Matrix<Y> Matrix<Y>::operator*=(const Y& rhs) {
    Cuda.cmultiply(this->_data, rhs, this->size());
    return *this;
}
template <typename Y>
Matrix<Y> Matrix<Y>::operator/=(const Y& rhs) {
    Cuda.cdivide(this->_data, rhs, this->size());
    return *this;
}


template <typename Y>
Matrix<Y> Matrix<Y>::operator+(const Matrix<Y>& rhs) {
    Matrix<Y> temp(*this);
    return temp += rhs;
}

template <typename Y>
Matrix<Y> Matrix<Y>::operator-(const Matrix<Y>& rhs) {
    Matrix<Y> temp(*this);
    return temp -= rhs;
}

template <typename Y>
Matrix<Y> Matrix<Y>::operator*(const Matrix<Y>& rhs) {
    Matrix<Y> temp(*this);
    return temp *= rhs;
}

template <typename Y>
Matrix<Y> Matrix<Y>::operator*(const Y& rhs) {
    Matrix<Y> temp(*this);
    return temp *= rhs;
}

template <typename Y>
Matrix<Y> Matrix<Y>::operator/(const Y& rhs) {
    Matrix<Y> temp(*this);
    return temp /= rhs;
}





///////////////////////////////////////////////////////MATEMATYCZNE///////////////////////////////////////////////////////////
template <typename Y>
Matrix<Y> Matrix<Y>::dot(const Matrix<Y>& rhs) {
    //variables
    cudaError_t cudaStatus;
    unsigned int total_size_a = size() * sizeof(Y);
    unsigned int total_size_b = rhs.size() * sizeof(Y);
    unsigned int total_size_c = this->_rows * rhs._cols * sizeof(Y);

    Y* host_a = this->_data;
    Y* host_b = rhs._data;
    Y* host_c = new Y[total_size_c];
    yeet m = this->_rows;
    yeet n = this->_cols;
    yeet k = rhs._cols;

    Y* dev_a;
    Y* dev_b;
    Y* dev_c;

    //set the device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) cout << "Setting device failed" << endl;

    //allocate memory
    cudaStatus = cudaMalloc((void**)&dev_a, total_size_a);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed [a]" << endl;
    cudaStatus = cudaMalloc((void**)&dev_b, total_size_b);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed [b]" << endl;
    cudaStatus = cudaMalloc((void**)&dev_c, total_size_c);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed [c]" << endl;

    //copy vectors to GPU
    cudaStatus = cudaMemcpy(dev_a, host_a, total_size_a, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) cout << "Copying to device failed [a]" << endl;
    cudaStatus = cudaMemcpy(dev_b, host_b, total_size_b, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) cout << "Copying to device failed [b]" << endl;

    //launch kernel
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid((k + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dotKernel << <grid, threads >> > (dev_a, dev_b, dev_c, m, n, k);

    //check if errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) cout << "Launching kernel failed [dot]" << endl;


    //synchronize devices
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) cout << "Synchronizing failed" << endl;

    //copy output to host
    cudaStatus = cudaMemcpy(host_c, dev_c, total_size_c, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) cout << "Copying to host failed\n" << cudaStatus << endl;

    //free memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    //reset the device
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) cout << "Resetting device failed" << endl;

    //create output Matrix
    Matrix<Y> C(this->_rows, rhs._cols, host_c);

    delete[] host_c;

    return C;
}

template <typename Y>
Matrix<Y> Matrix<Y>::T() {
    //variables
    cudaError_t cudaStatus;
    unsigned int total_size = size() * sizeof(Y);

    Y* host_a = this->_data;
    Y* host_c = new Y[size()];

    Y* dev_a;
    Y* dev_c;

    //set the device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) cout << "Setting device failed" << endl;

    //allocate memory
    cudaStatus = cudaMalloc((void**)&dev_a, total_size);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed [a]" << endl;
    cudaStatus = cudaMalloc((void**)&dev_c, total_size);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed [c]" << endl;

    //copy vectors to GPU
    cudaStatus = cudaMemcpy(dev_a, host_a, total_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) cout << "Copying to device failed [a]" << endl;

    //launch kernel
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid((size() - 1) / THREADS_PER_BLOCK + 1, (size() - 1) / THREADS_PER_BLOCK + 1);
    transposeKernel << <grid,threads>> > (dev_a, dev_c, this->_rows,this->_cols);

    //check if errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) cout << "Launching kernel failed [dot]" << endl;


    //synchronize devices
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) cout << "Synchronizing failed" << endl;

    //copy output to host
    cudaStatus = cudaMemcpy(host_c, dev_c, total_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) cout << "Copying to host failed\n" << cudaStatus << endl;

    //free memory
    cudaFree(dev_a);
    cudaFree(dev_c);

    //reset the device
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) cout << "Resetting device failed" << endl;

    //create output Matrix
    Matrix<Y> C(this->_cols, this->_rows, host_c);

    delete[] host_c;

    return C;
}

template <typename Y>
Matrix<Y> Matrix<Y>::sigmoid() {
    Matrix<Y> output(*this);
    Cuda.sigmoid(output._data, size());
    return output;
}

template <typename Y>
Matrix<Y> Matrix<Y>::sigmoid_derivative() {
    Matrix<Y> output(*this);
    Cuda.sigmoid_derivative(output._data, size());
    return output;
}

template <typename Y>
Matrix<Y> Matrix<Y>::square() {
    Matrix<Y> output(*this);
    Cuda.square(output._data, size());
    return output;
}

template <typename Y>
double Matrix<Y>::mean() {   ///this works on CPU only //todo
    double output = 0.0;
    double count = this->size();
    double suma = 0.0;

    double* arr = this->getArray();

    for (yeet i = 0; i < count; i++)
        suma += arr[i];

    output = (suma / (double)count);

    delete[] arr;

    return output;
}

#endif