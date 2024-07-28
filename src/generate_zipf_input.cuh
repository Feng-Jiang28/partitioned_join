#ifndef GENERATE_ZIPF_INPUT_CUH
#define GENERATE_ZIPF_INPUT_CUH

// zipf distribution generator 
// https://github.com/psiul/ICDE2019-GPU-Join/blob/42bcdd6e4bbaf8da1a9a640d87f9351e4efb9738/src/generator_ETHZ.cu 

#include <cassert>
#include <cstdlib>
#include <string> 
#include <fstream>

/**
 * Create an alphabet, an array of size @a size with randomly
 * permuted values 0..size-1.
 *
 * @param size alphabet size
 * @return an <code>item_t</code> array with @a size elements;
 *         contains values 0..size-1 in a random permutation; the
 *         return value is malloc'ed, don't forget to free it afterward.
 */
 static uint32_t *gen_alphabet(unsigned int size) {
	uint32_t *alphabet;

	/* allocate */
	alphabet = (uint32_t *) malloc(size * sizeof(*alphabet));
	assert(alphabet);

	/* populate */
	for (unsigned int i = 0; i < size; i++)
		alphabet[i] = i + 1; /* don't let 0 be in the alphabet */

	/* permute */
	for (unsigned int i = size - 1; i > 0; i--) {
		unsigned int k = (unsigned long) i * rand() / RAND_MAX;
		unsigned int tmp;

		tmp = alphabet[i];
		alphabet[i] = alphabet[k];
		alphabet[k] = tmp;
	}

	return alphabet;
}

/**
 * Generate a lookup table with the cumulated density function
 *
 * (This is derived from code originally written by Rene Mueller.)
 */
 static double *gen_zipf_lut(double zipf_factor, unsigned int alphabet_size) {
	double scaling_factor;
	double sum;

	double *lut; /**< return value */

	lut = (double *) malloc(alphabet_size * sizeof(*lut));
	assert(lut);

	/*
	 * Compute scaling factor such that
	 *
	 *   sum (lut[i], i=1..alphabet_size) = 1.0
	 *
	 */
	scaling_factor = 0.0;
	for (unsigned int i = 1; i <= alphabet_size; i++)
		scaling_factor += 1.0 / pow(i, zipf_factor);

	/*
	 * Generate the lookup table
	 */
	sum = 0.0;
	for (unsigned int i = 1; i <= alphabet_size; i++) {
		sum += 1.0 / pow(i, zipf_factor);
		lut[i - 1] = sum / scaling_factor;
	}

	return lut;
}

/**
 * Generate a stream with Zipf-distributed content.
 */
 template<typename key_type>
 void gen_zipf(uint64_t stream_size, unsigned int alphabet_size, double zipf_factor, key_type *ret) {
	//uint64_t i;
	/* prepare stuff for Zipf generation */
	uint32_t *alphabet = gen_alphabet(alphabet_size);
	assert(alphabet);

	double *lut = gen_zipf_lut(zipf_factor, alphabet_size);
	assert(lut);

	// uint32_t seeds[64];

	// for (int i = 0; i < 64; i++)
	// 	seeds[i] = rand();

	for (uint64_t i = 0; i < stream_size; i++) {
		// if (i % 1000000 == 0)
		// 	printf("live %" PRId64 "\n", i / 1000000);

		/* take random number */
		double r;

		r = ((double) (rand())) / RAND_MAX;

		/* binary search in lookup table to determine item */
		unsigned int left = 0;
		unsigned int right = alphabet_size - 1;
		unsigned int m; /* middle between left and right */
		unsigned int pos; /* position to take */

		if (lut[0] >= r)
			pos = 0;
		else {
			while (right - left > 1) {
				m = (left + right) / 2;

				if (lut[m] < r)
					left = m;
				else
					right = m;
			}

			pos = right;
		}

		ret[i] = static_cast<key_type>(alphabet[pos]);
	}

	free(lut);
	free(alphabet);
}

template<typename key_type>
int readFromFile(const char * filename, key_type *relation, uint64_t num_tuples) {
	char path[100];
	sprintf(path, "%s", filename);
	FILE *fp = fopen(path, "rb");

	if (!fp) return 1;

	printf("    Reading file %s ... ", path);
	fflush(stdout);

	int num_rows = fread(relation, sizeof(int), num_tuples, fp);
	printf("    %d elements read\n", num_rows);

	fclose(fp);
	return 0;
}

template<typename key_type>
void load_tpch_q4_data(
	std::string lfile, std::string ofile, const int64_t lnum_rows, const int64_t onum_rows, 
	key_type* l_orderkey, key_type* o_orderkey, int* l_payload
) {  
    constexpr int MAX_ELEM_CHAR = 25; 
	// load lineitem table
	std::cout << "Loading lineitem" << std::endl;
    
	std::ifstream infile(lfile);
	std::string line;
  
	int64_t line_count = 0;
	while (std::getline(infile, line) && line_count < lnum_rows) {  
		int idx[1] = {0};
		int elem_idx = 0;
		int elem_char_idx = 0;
		int line_char_idx = 0;
	
		char element[MAX_ELEM_CHAR];
	
		while ('|' != line[line_char_idx])
			element[elem_char_idx++] = line[line_char_idx++];
		line_char_idx++;
	
		for(int i = 0; i < 1; i++) {
	
			while (elem_idx < idx[i]) {
				elem_char_idx = 0;
	
				while ('|' != line[line_char_idx])
					element[elem_char_idx++] = line[line_char_idx++];
				line_char_idx++;
		
				elem_idx++;
			}
	
			element[elem_char_idx] = '\0';
			if (0 == i)
				l_orderkey[line_count] = static_cast<key_type>(atoi(element));
		}
	
		line_count++;
	}
  
	if (line_count < lnum_rows) {
		for (int64_t i = line_count; i < lnum_rows; i++) {
			int64_t copy_idx = i % line_count;
			l_orderkey[i] = l_orderkey[copy_idx];
		}
	}
    
	// load order table
	std::cout << "Loading orders" << std::endl;  
	std::ifstream orfile(ofile);
  
	line_count = 0;
	while (std::getline(orfile, line) && line_count < onum_rows) {
		int idx[2] = {0, 5};
		int elem_idx = 0;
		int elem_char_idx = 0;
		int line_char_idx = 0;
	
		char element[MAX_ELEM_CHAR];
	
		while ('|' != line[line_char_idx])
			element[elem_char_idx++] = line[line_char_idx++];
		line_char_idx++;
	
		for(int i = 0; i < 2; i++) {
	
			while (elem_idx < idx[i]) {
				elem_char_idx = 0;
		
				while ('|' != line[line_char_idx])
					element[elem_char_idx++] = line[line_char_idx++];
				line_char_idx++;
		
				elem_idx++;
			}
			element[elem_char_idx] = '\0';
			if (0 == i)
				o_orderkey[line_count] = static_cast<key_type>(atoi(element));
			if (1 == i) {
				l_payload[line_count] = static_cast<int>(element[0]);
			}
		}
	
		line_count++;
	}
  
	if (line_count < onum_rows) {
		for (int64_t i = line_count; i < onum_rows; i++) {
			int64_t copy_idx = i % line_count;
			o_orderkey[i] = o_orderkey[copy_idx];
		}
	}
  
	std::cout << std::endl;
}

#endif // GENERATE_ZIPF_INPUT_CUH