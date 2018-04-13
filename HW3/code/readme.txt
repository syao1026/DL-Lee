my_utils.py -- read training data(.csv file), and some preprocessing like normalization

vgg16_train.py -- inlucde 2 net for training

test_vgg16.py -- test the result using network in  vgg16_train.py, what i use is vgg16-like network
                  but not vgg16 since my training data is 48*48 which is too small for that.
                
                  but maybe can upsampling to image first, then try vgg16?, not sure about the result

                  Besides, this network is quite simple since when i put more neurals in hidden layers,
                  my computer told me out of memory, sad story... _(:???)_

Img-Sentiment-cls.py and Img-Sentiment_cls-keras.py -- you can ignore these two files, the aboved three files 
                                                      are more organized version of these two files. :)


