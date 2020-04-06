from aspect_extraction.apply_extraction import apply_extraction

def extract_aspects(reviews,nlp,sid, text_column, review_id, product_id):

    print("Entering Apply function!")
    aspect_list = reviews.apply(lambda row: apply_extraction(row,nlp,sid, 
                                                             text_column, 
                                                             review_id, 
                                                             product_id), axis=1) #going through all the rows in the dataframe
        
    return aspect_list