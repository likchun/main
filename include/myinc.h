/**
 * @file myinc.h
 * @author likchun@outlook.com
 * @brief my includes
 * @version 0.2
 * @date 2022-03-12
 * 
 */

#ifndef MY_INC_H
#define MY_INC_H

#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <string>
#include <vector>
#include <tuple>
#include <chrono>

namespace myinc
{
	/**
	 * Return the average of values in a C++ vector.
	 *
	 * @tparam T the type of data stored in the vector
	 * @param[in] vect vector of T
	 * @param[in] begin_pos the number of elements to be removed from the begining of the vector during calculation
	 * @param[in] end_pos the number of elements to be removed from the end of the vector during calculation
	 * @returns average of values in vector
	 */
	template<typename T>
	double average_vect(std::vector<T> const& vect, size_t begin_pos=0, size_t end_pos=0)
	{
		if (vect.empty()) {
			return 0;
		}
		if (begin_pos+end_pos >= vect.size()) {
			char* msg = nullptr;
			sprintf(msg, "ArgumentValueError: when calling function %s(), 'start_pos+end_pos' greater than vector length.", __func__);
			std::invalid_argument err(msg);
			std::cerr << err.what() << '\n';
			throw err;
		}
		auto const length = static_cast<T>(vect.size()-begin_pos-end_pos);
		return std::accumulate(vect.begin()+begin_pos, vect.end()-end_pos, 0.0) / length;
	}

	/**
	 * Multiply two C++ vectors, element wise.
	 *
	 * @tparam T the type of data stored in the vector
	 * @param[in] vect1 vector of T to be multiplied
	 * @param[in] vect2 vector of T to be multiplied
	 * @param[in] lag_i multiply the (i+lag_i)-th element of vect1 to the i-th element of vect2, resultant vector will be (lag_i) shorther
	 * @returns product vector from muliplication
	 */
	template<typename T>
	std::vector<T> multiply_vect(std::vector<T>& vect1, std::vector<T>& vect2, size_t lag_i=0)
	{
		if (vect1.size() == vect2.size()) {
			if (lag_i == 0) {
				std::vector<T> product(vect1.size());
				for (size_t i = 0; i < vect1.size(); i++) {
					product[i] = vect1[i] * vect2[i];
				}
				return product;
			} else {
				std::vector<T> product(vect1.size()-lag_i);
				for (size_t i = 0; i < vect1.size()-lag_i; i++) {
					product[i] = vect1[i+lag_i] * vect2[i];
				}
				return product;
			}
		} else {
			char* msg = nullptr;
			sprintf(msg, "ArgumentValueError: when calling function %s(), the two input vectors are of different sizes.", __func__);
			std::invalid_argument err(msg);
			std::cerr << err.what() << '\n';
			throw err;
		}
	}

	/**
	 * Flatten a 2D C++ vector.
	 *
	 * @tparam T the type of data stored in the vector
	 * @param[in] vect2d 2D vector whose rows are of various sizes
	 * @returns flatted vector
	 */
	template<typename T>
	std::vector<T> flatten_vect(const std::vector<std::vector<T>> &vect2d)
	{   
		std::vector<T> result;
		for (const auto &v : vect2d) {
			result.insert(result.end(), v.begin(), v.end());
		}
		return result;
	}

	/**
	 * Read matrix from a file, elements in rows are tab `\\t` separated.
	 *
	 * @tparam T the type of data stored in the matrix
	 * @param[in] vect2d 2D vector or matrix
	 * @param[in] filename file name of the input file, including extention
	 * @param[in] delimiter delimiter char separating the elements in a row
	 * @returns 0 if succeed
	 */
	template<typename T>
	int fread_dense_matrix(std::vector<std::vector<T>>& vect2d, std::string filename, char delimiter)
	{
		vect2d = std::vector<std::vector<T>>();
		std::vector<T> row_buffer;
		std::string line, element;
		std::ifstream fs(filename);
		if (fs.is_open()) {
			while(getline(fs, line, '\n')) {
				std::stringstream ss(line);
				while(getline(ss, element, delimiter)) {
					row_buffer.push_back(stof(element));
				}
				vect2d.push_back(row_buffer);
				row_buffer.clear();
			}
			fs.close();
		} else {
			char* msg = nullptr;
			sprintf(msg, "FileAccessError: when calling function %s(), [%s] cannot be accessed.", __func__, filename.c_str());
			std::invalid_argument err(msg);
			std::cerr << err.what() << '\n';
			throw err;
		}
		return EXIT_SUCCESS;
	}

	/**
	 * Read matrix from a file, elements in rows are tab `\\t` separated.
	 *
	 * @tparam T the type of data stored in the matrix
	 * @param[in] vect2d 2D vector or matrix
	 * @param[in] filename file name of the input file, including extention
	 * @param[in] delimiter delimiter char separating the elements in a row
	 * @returns 0 if succeed
	 */
	template<typename T>
	int fread_dense_matrix(std::vector<std::vector<T>>& vect2d, std::filesystem::path filename, char delimiter)
	{
		vect2d = std::vector<std::vector<T>>();
		std::vector<T> row_buffer;
		std::string line, element;
		std::ifstream fs(filename);
		if (fs.is_open()) {
			while(getline(fs, line, '\n')) {
				std::stringstream ss(line);
				while(getline(ss, element, delimiter)) {
					row_buffer.push_back(stof(element));
				}
				vect2d.push_back(row_buffer);
				row_buffer.clear();
			}
			fs.close();
		} else {
			char* msg = nullptr;
			sprintf(msg, "FileAccessError: when calling function %s(), [%s] cannot be accessed.", __func__, filename.string().c_str());
			std::invalid_argument err(msg);
			std::cerr << err.what() << '\n';
			throw err;
		}
		return EXIT_SUCCESS;
	}

	/**
	 * Write matrix into a file, elements in rows are tab `\\t` separated
	 *
	 * @tparam T the type of data stored in the matrix
	 * @param[in] vect2d 2D vector or matrix
	 * @param[in] filename file name of the output file, including extention
	 * @returns 0 if succeed
	 */
	template<typename T>
	int fwrite_dense_matrix(std::vector<std::vector<T>>& vect2d, std::string filename)
	{
		std::ofstream fs(filename, std::ios::out);
		if (fs.is_open()) {
			fs << vect2d[0][0];
			for (size_t j = 1; j < vect2d[0].size(); j++) {
				fs << '\t' << vect2d[0][j];
			}
			for (size_t i = 1; i < vect2d.size(); i++) {
				fs << '\n';
				fs << vect2d[i][0];
				for (size_t j = 1; j < vect2d[i].size(); j++) {
					fs << '\t' << vect2d[i][j];
				}
			}
			fs.close();
		} else {
			char* msg = nullptr;
			sprintf(msg, "FileAccessError: when calling function %s(), [%s] cannot be accessed.", __func__, filename.c_str());
			std::invalid_argument err(msg);
			std::cerr << err.what() << '\n';
			throw err;
		}
		return EXIT_SUCCESS;
	}

	/**
	 * Write matrix into a file, elements in rows are tab `\\t` separated
	 *
	 * @tparam T the type of data stored in the matrix
	 * @param[in] vect2d 2D vector or matrix
	 * @param[in] filename file name of the output file, including extention
	 * @returns 0 if succeed
	 */
	template<typename T>
	int fwrite_dense_matrix(std::vector<std::vector<T>>& vect2d, std::filesystem::path filename)
	{
		std::ofstream fs(filename, std::ios::out);
		if (fs.is_open()) {
			fs << vect2d[0][0];
			for (size_t j = 1; j < vect2d[0].size(); j++) {
				fs << '\t' << vect2d[0][j];
			}
			for (size_t i = 1; i < vect2d.size(); i++) {
				fs << '\n';
				fs << vect2d[i][0];
				for (size_t j = 1; j < vect2d[i].size(); j++) {
					fs << '\t' << vect2d[i][j];
				}
			}
			fs.close();
		} else {
			char* msg = nullptr;
			sprintf(msg, "FileAccessError: when calling function %s(), [%s] cannot be accessed.", __func__, filename.string().c_str());
			std::invalid_argument err(msg);
			std::cerr << err.what() << '\n';
			throw err;
		}
		return EXIT_SUCCESS;
	}

	/**
	 * Write matrix into a file, each row represents an element with format: `row i` `col j` `value`, whitespace separated.
	 * Effecient for sparse matrix.
	 *
	 * @tparam T the type of data stored in the matrix
	 * @param[in] vect2d 2D vector or matrix
	 * @param[in] filename file name of the output file, including extention
	 * @returns 0 if succeed
	 */
	template<typename T>
	int fwrite_sparse_matrix(std::vector<std::vector<T>>& vect2d, std::string filename)
	{
		std::ofstream fs(filename, std::ios::out);
		if (fs.is_open()) {
			for (int i = 1; i < vect2d.size(); i++) {
				for (int j = 1; j < vect2d[i].size(); j++) {
					if (vect2d[i][j] != 0) {
						fs << i << ' ' << j << ' ' << vect2d[i][j] << '\n';
					}
				}
			}
			fs.close();
		} else {
			char* msg = nullptr;
			sprintf(msg, "FileAccessError: when calling function %s(), [%s] cannot be accessed.", __func__, filename.c_str());
			std::invalid_argument err(msg);
			std::cerr << err.what() << '\n';
			throw err;
		}
		return EXIT_SUCCESS;
	}

	/**
	 * Write matrix into a file, each row represents an element with format: `row i` `col j` `value`, whitespace separated.
	 * Effecient for sparse matrix.
	 *
	 * @tparam T the type of data stored in the matrix
	 * @param[in] vect2d 2D vector or matrix
	 * @param[in] filename file name of the output file, including extention
	 * @returns 0 if succeed
	 */
	template<typename T>
	int fwrite_sparse_matrix(std::vector<std::vector<T>>& vect2d, std::filesystem::path filename)
	{
		std::ofstream fs(filename, std::ios::out);
		if (fs.is_open()) {
			for (int i = 1; i < vect2d.size(); i++) {
				for (int j = 1; j < vect2d[i].size(); j++) {
					if (vect2d[i][j] != 0) {
						fs << i << ' ' << j << ' ' << vect2d[i][j] << '\n';
					}
				}
			}
			fs.close();
		} else {
			char* msg = nullptr;
			sprintf(msg, "FileAccessError: when calling function %s(), [%s] cannot be accessed.", __func__, filename.string().c_str());
			std::invalid_argument err(msg);
			std::cerr << err.what() << '\n';
			throw err;
		}
		return EXIT_SUCCESS;
	}

	/**
	 * @brief Alternative output streambuffer, output to the standard output device and a log file. Basically, std::cout and fstream combined.
	 * Initialization 1: OstreamLog outlog("log_filename.txt"); outlog << "your message";
	 * or
	 * Initialization 2: OstreamLog outlog; outlog.open("log_filename.txt"); outlog << "your message";
	 * 
	 */
	class OstreamLog
	{
	public:
		OstreamLog() {};
		OstreamLog(std::string log_fpath) : log_fstream(log_fpath, std::ios::app) {};
		OstreamLog(std::filesystem::path log_fpath) : log_fstream(log_fpath, std::ios::app) {};
		OstreamLog(std::string log_fpath, std::ios_base::openmode open_mode) : log_fstream(log_fpath, open_mode) {};
		OstreamLog(std::filesystem::path log_fpath, std::ios_base::openmode open_mode) : log_fstream(log_fpath, open_mode) {};

		// Use it like cout. OstreamLog outlog; outlog << "message";
		template<typename T> OstreamLog& operator << (const T& msg)		// for regular output of variables and stuff
		{
			std::cout << msg;
			log_fstream << msg;
			return *this;
		}
		typedef std::ostream& (*stream_function)(std::ostream&);		// for manipulators like std::endl
		// Use it like cout. OstreamLog outlog; outlog << "message";
		OstreamLog& operator << (stream_function func)
		{
			func(std::cout);
			func(log_fstream);
			return *this;
		}
		/**
		 * @brief Initialize OstreamLog. Open file to output stream buffer. Default open mode: ios::app.
		 * 
		 * @param log_fpath path of file to output stream buffer.
		 */
		void open(std::string log_fpath)
		{
			log_fstream.open(log_fpath, std::ios::app);
			if (!log_fstream.is_open()) { std::cerr << "FileAccessError: log file [" << log_fpath << "] cannot be opened"; }
		}
		/**
		 * @brief Initialize OstreamLog. Open file to output stream buffer. Default open mode: ios::app.
		 * 
		 * @param log_fpath path of file to output stream buffer.
		 */
		void open(std::filesystem::path log_fpath)
		{
			log_fstream.open(log_fpath, std::ios::app);
			if (!log_fstream.is_open()) { std::cerr << "FileAccessError: log file [" << log_fpath << "] cannot be opened"; }
		}
		/**
		 * @brief Initialize OstreamLog. Open file to output stream buffer.
		 * 
		 * @param log_fpath path of file to output stream buffer.
		 * @param open_mode open mode, e.g., std::ios::app, std::ios::trunc, etc.
		 */
		void open(std::string log_fpath, std::ios_base::openmode open_mode)
		{
			log_fstream.open(log_fpath, open_mode);
			if (!log_fstream.is_open()) { std::cerr << "FileAccessError: log file [" << log_fpath << "] cannot be opened"; }
		}
		/**
		 * @brief Initialize OstreamLog. Open file to output stream buffer.
		 * 
		 * @param log_fpath path of file to output stream buffer.
		 * @param open_mode open mode, e.g., std::ios::app, std::ios::trunc, etc.
		 */
		void open(std::filesystem::path log_fpath, std::ios_base::openmode open_mode)
		{
			log_fstream.open(log_fpath, open_mode);
			if (!log_fstream.is_open()) { std::cerr << "FileAccessError: log file [" << log_fpath << "] cannot be opened"; }
		}
	private:
		std::ofstream log_fstream;
	} outlog;

	class Chronometer
	{
	public:
		Chronometer () {
			isActivated = false;
			isPaused	= false;
			pause_duration		= 0;
			lap_pause_duration	= 0;
		}
		// Return current time (in customized format)
		std::tuple<std::string, std::string> report_time(const char* fmt="%y-%m-%d_%H-%M-%S") {
			start_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			ctime_s(ctime_buf1, sizeof(ctime_buf1), &start_t);
			strftime(ctime_buf2, sizeof(ctime_buf2), fmt, localtime(&start_t));
			return {std::string(ctime_buf1), std::string(ctime_buf2)};
		}
		// Start the stopwatch, get elapsed time using stopwatch_elapsed_t() or stopwatch_lap()
		void stopwatch_begin() {
			isActivated = true;
			beg = std::chrono::steady_clock::now();
			lap_beg = beg;
		}
		// Return the time in milliseconds (ms)) passed since calling Chronometer.stopwatch_start()
		// or -1 if begin() has not been called
		double stopwatch_elapsed_t() {
			if (isActivated) {
				return (double)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - beg).count() - pause_duration;
			} else {
				return -1;
			}
		}
		// Return the time elapsed in millisecond (ms) since the last time calling Chronometer.stopwatch_lap(),
		// then start a new lap. Return -1 if Chronometer.stopwatch_start() has not been called
		double stopwatch_lap() {
			if (isActivated) {
				lap_last = lap_beg;
				lap_beg = std::chrono::steady_clock::now();
				return (double)std::chrono::duration_cast<std::chrono::milliseconds>(lap_beg - lap_last).count() - lap_pause_duration;
			} else {
				return -1;
			}
		}
		void pause() {
			if (isActivated) {
				pau_beg = std::chrono::steady_clock::now();
				isPaused = true;
			}
		}
		void resume() {
			if (isActivated && isPaused) {
				pau_fin = std::chrono::steady_clock::now();
				pause_duration += (double)std::chrono::duration_cast<std::chrono::milliseconds>(pau_fin - pau_beg).count();
				lap_pause_duration = (double)std::chrono::duration_cast<std::chrono::milliseconds>(pau_fin - pau_beg).count();
				isPaused = false;
			}
		}
		void destroy() { this->~Chronometer(); }
		~Chronometer() {}
	private:
		time_t start_t;
		std::chrono::steady_clock::time_point beg, lap_beg, lap_last, pau_beg, pau_fin;
		bool isActivated, isPaused;
		double pause_duration, lap_pause_duration;
		char ctime_buf1[30], ctime_buf2[30];
	} timer;
}; // namespace myinc

#endif