# Utility functions for tokenization 
def apply_tokenizer(tokenizer, text, sum, max_len):
	# TODO: Complete this function 
	# Concatenate text and summary and in-between add a special token 
	# <BOS> is beginning of sentence 
	# <EOS> is end of sentence
	  ts =  text + ' <SUMMARY> ' + sum
    #encodings = tokenizer('<BOS> ' + ts + ' <EOS>', truncation=True, max_length=max_len, padding="max_length")
	return tokenizer('<BOS>'  '<EOS>', truncation=True, max_length=max_len, padding="max_length")

def tokenize_text(tokenizer, df, max_len):
    df['encodings'] = df.apply(lambda x : apply_tokenizer(tokenizer, x['text'], x['summary'], max_len), axis=1)
    #encodings is dict type: contains input_ids and attention_mas
    
    del df['text']
    del df['summary']

    return df

def tokenize_dataset(tokenizer, train, val, test, max_len):
	train = tokenize_text(tokenizer, train, max_len)
	val = tokenize_text(tokenizer, val, max_len)
	
	return train, val, test
