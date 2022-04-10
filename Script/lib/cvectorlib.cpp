#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

inline long long get_file_size(string file_name)
{
    struct stat64 stat_buf;
    int rc = stat64(file_name.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

inline void stream_info(istream& stream)
{
    cout << "Good: " << stream.good() <<
            " | Eof: " << stream.eof() <<
            " | Bad: " << stream.bad() <<
            " | Fail: " << stream.fail() << endl;
}

extern "C"
{
    int read_from_binary_int(int node_index, int matrix_size, vector<int>* vect, const char* file_name)
    {
        ifstream myfile(file_name, ios::in | ios::binary);
        if (myfile.is_open()) {
            struct stat64 stat_buf;
            string file_name_str = file_name;
            int rc = stat64(file_name_str.c_str(), &stat_buf);
            long long file_size = (rc == 0) ? stat_buf.st_size : -1;

            vector<int> buffer;
            int content_buf;
            myfile.seekg(node_index*sizeof(int), ios::beg);
            myfile.read(reinterpret_cast<char*>(&content_buf), sizeof(int));
            buffer.push_back(content_buf);
            while (myfile.eof() == 0) {
                myfile.seekg((matrix_size-1)*sizeof(int), ios::cur);
                myfile.read(reinterpret_cast<char*>(&content_buf), sizeof(int));
                buffer.push_back(content_buf);
            }
            buffer.pop_back();
            myfile.close();
            *vect = buffer;
            vector<int>().swap(buffer);
            return EXIT_SUCCESS;
        } else {
            return EXIT_FAILURE;
        }
    }
    vector<int>* new_vector_int() {
        return new vector<int>;
    }
    void delete_vector_int(vector<int>* vect) {
        // cout << "destructor called in C++ for " << vect << endl;
        delete vect;
    }
    int vector_size_int(vector<int>* vect) {
        return vect->size();
    }
    int vector_get_int(vector<int>* vect, int i) {
        return vect->at(i);
    }
    void vector_set_int(vector<int>* vect, int i, int x) {
        vect->at(i) = x;
    }
    void vector_push_back_int(vector<int>* vect, int i) {
        vect->push_back(i);
    }
    void vector_pop_back_int(vector<int>* vect) {
        vect->pop_back();
    }

    int read_from_binary_fl(vector<float>* vect, const char* file_name, int node_index, int matrix_size)
    {
        ifstream myfile(file_name, ios::in | ios::binary);
        if (myfile.is_open()) {
            struct stat64 stat_buf;
            string file_name_str = file_name;
            int rc = stat64(file_name_str.c_str(), &stat_buf);
            long long file_size = (rc == 0) ? stat_buf.st_size : -1;

            vector<float> buffer;
            float content_buf;
            myfile.seekg(node_index*sizeof(float), ios::beg);
            myfile.read(reinterpret_cast<char*>(&content_buf), sizeof(float));
            buffer.push_back(content_buf);
            while (myfile.eof() == 0) {
                myfile.seekg((matrix_size-1)*sizeof(float), ios::cur);
                myfile.read(reinterpret_cast<char*>(&content_buf), sizeof(float));
                buffer.push_back(content_buf);
            }
            buffer.pop_back();
            myfile.close();
            *vect = buffer;
            vector<float>().swap(buffer);
            return EXIT_SUCCESS;
        } else {
            return EXIT_FAILURE;
        }

        // Legacy code:
        // string line;
        // ifstream file_stream(file_name, ios::in | ios::binary);
        // if (file_stream.is_open()) {
        //     file_stream.seekg(0, ios::end);
        //     int file_size = file_stream.tellg();
        //     vector<float> buffer(file_size/sizeof(float));
        //     file_stream.seekg(0, ios::beg);
        //     file_stream.read(reinterpret_cast<char*>(&buffer[0]), file_size);
        //     file_stream.close();
        //     *vect = buffer;
        //     return EXIT_SUCCESS;
        // } else {
        //     return EXIT_FAILURE;
        // }
    }
    vector<float>* new_vector_fl() {
        return new vector<float>;
    }
    void delete_vector_fl(vector<float>* vect) {
        // cout << "destructor called in C++ for " << vect << endl;
        delete vect;
    }
    int vector_size_fl(vector<float>* vect) {
        return vect->size();
    }
    float vector_get_fl(vector<float>* vect, int i) {
        return vect->at(i);
    }
    void vector_set_fl(vector<float>* vect, int i, float x) {
        vect->at(i) = x;
    }
    void vector_push_back_fl(vector<float>* vect, float i) {
        vect->push_back(i);
    }
    void vector_pop_back_fl(vector<int>* vect) {
        vect->pop_back();
    }

    int read_from_binary_db(int node_index, int matrix_size, vector<double>* vect, const char* file_name)
    {
        ifstream myfile(file_name, ios::in | ios::binary);
        if (myfile.is_open()) {
            struct stat64 stat_buf;
            string file_name_str = file_name;
            int rc = stat64(file_name_str.c_str(), &stat_buf);
            long long file_size = (rc == 0) ? stat_buf.st_size : -1;

            vector<double> buffer;
            double content_buf;
            myfile.seekg(node_index*sizeof(double), ios::beg);
            myfile.read(reinterpret_cast<char*>(&content_buf), sizeof(double));
            buffer.push_back(content_buf);
            while (myfile.eof() == 0) {
                myfile.seekg((matrix_size-1)*sizeof(double), ios::cur);
                myfile.read(reinterpret_cast<char*>(&content_buf), sizeof(double));
                buffer.push_back(content_buf);
            }
            buffer.pop_back();
            myfile.close();
            *vect = buffer;
            vector<double>().swap(buffer);
            return EXIT_SUCCESS;
        } else {
            return EXIT_FAILURE;
        }
    }
    vector<double>* new_vector_db() {
        return new vector<double>;
    }
    void delete_vector_db(vector<double>* vect) {
        // cout << "destructor called in C++ for " << vect << endl;
        delete vect;
    }
    int vector_size_db(vector<double>* vect) {
        return vect->size();
    }
    double vector_get_db(vector<double>* vect, int i) {
        return vect->at(i);
    }
    void vector_set_db(vector<double>* vect, int i, double x) {
        vect->at(i) = x;
    }
    void vector_push_back_db(vector<double>* vect, double i) {
        vect->push_back(i);
    }
    void vector_pop_back_db(vector<int>* vect) {
        vect->pop_back();
    }
}