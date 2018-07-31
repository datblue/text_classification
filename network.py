from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Bidirectional


def building_network(embedding_size, num_lstm_layer, num_hidden_node,
                     dropout, time_step, output_lenght):
    model = Sequential()

    # model.add(LSTM(num_hidden_node, return_sequences=True, dropout=dropout,
    #                input_shape=(time_step, embedding_size)))
    #
    # for i in xrange(num_lstm_layer-1):
    #     model.add(LSTM(num_hidden_node, return_sequences=True, dropout=dropout))

    model.add(Bidirectional(LSTM(num_hidden_node, return_sequences=True,
                                 recurrent_dropout=dropout,
                                 input_shape=(time_step, embedding_size))))

    for i in xrange(num_lstm_layer-1):
        model.add(Bidirectional(LSTM(num_hidden_node, return_sequences=True,
                                     recurrent_dropout=dropout),
                                merge_mode='concat'))

    model.add(Flatten())

    model.add(Dense(output_lenght, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model