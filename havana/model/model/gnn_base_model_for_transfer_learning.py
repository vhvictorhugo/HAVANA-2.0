import tensorflow as tf
from spektral.layers.convolutional import ARMAConv, GATConv
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class GNNUS_BaseModel:
    def __init__(self, params):
        self.max_size_matrices = params["max_size_matrices"]
        self.max_size_sequence = params["max_size_sequence"]
        self.num_classes = params["num_classes"]
        self.features_num_columns = params["features_num_columns"]
        self.share_weights = False
        self.dropout_skip = params["dropout_skip"]
        self.dropout = params["dropout"]
        self.num_pois = 3
        self.baseline = params["baseline"]
        if not self.baseline:
            self.embeddings_dimension = params["embeddings_dimension"]

    def build(self, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)

        l2_reg = l2(5e-4)  # L2 regularization rate
        A_input = Input((self.max_size_matrices, self.max_size_matrices))
        A_week_input = Input((self.max_size_matrices, self.max_size_matrices))
        A_weekend_input = Input((self.max_size_matrices, self.max_size_matrices))
        Temporal_input = Input((self.max_size_matrices, self.features_num_columns))
        Temporal_week_input = Input((self.max_size_matrices, 24))
        Temporal_weekend_input = Input((self.max_size_matrices, 24))
        Distance_input = Input((self.max_size_matrices, self.max_size_matrices))
        Duration_input = Input((self.max_size_matrices, self.max_size_matrices))
        Location_time_input = Input((self.max_size_matrices, self.features_num_columns))
        Location_location_input = Input((self.max_size_matrices, self.max_size_matrices))
        if not self.baseline:
            User_embeddings_input = Input((self.max_size_matrices, self.embeddings_dimension))

        out_temporal = ARMAConv(
            20, kernel_regularizer=l2_reg, share_weights=self.share_weights, dropout_rate=self.dropout_skip
        )([Temporal_input, A_input])
        out_temporal = Dropout(self.dropout)(out_temporal)
        out_temporal = ARMAConv(20, kernel_regularizer=l2_reg)([out_temporal, A_input])

        out_week_temporal = ARMAConv(
            20, kernel_regularizer=l2_reg, share_weights=self.share_weights, dropout_rate=self.dropout_skip
        )([Temporal_week_input, A_week_input])
        out_week_temporal = Dropout(self.dropout)(out_week_temporal)
        out_week_temporal = ARMAConv(20, kernel_regularizer=l2_reg)([out_week_temporal, A_week_input])

        out_weekend_temporal = ARMAConv(
            20, kernel_regularizer=l2_reg, share_weights=self.share_weights, dropout_rate=self.dropout_skip
        )([Temporal_weekend_input, A_weekend_input])
        out_weekend_temporal = Dropout(self.dropout)(out_weekend_temporal)
        out_weekend_temporal = ARMAConv(20, kernel_regularizer=l2_reg)([out_weekend_temporal, A_weekend_input])

        out_distance = ARMAConv(
            20,
            kernel_regularizer=l2_reg,
        )([Distance_input, A_input])
        out_distance = Dropout(self.dropout)(out_distance)
        out_distance = ARMAConv(20, kernel_regularizer=l2_reg)([out_distance, A_input])

        out_duration = ARMAConv(
            20,
            kernel_regularizer=l2_reg,
        )([Duration_input, A_input])
        out_duration = Dropout(self.dropout)(out_duration)
        out_duration = ARMAConv(20, kernel_regularizer=l2_reg)([out_duration, A_input])

        out_location_location = ARMAConv(
            20,
            kernel_regularizer=l2_reg,
        )([Location_time_input, Location_location_input])
        out_location_location = Dropout(self.dropout)(out_location_location)
        out_location_location = ARMAConv(20, kernel_regularizer=l2_reg)(
            [out_location_location, Location_location_input]
        )

        out_location_time = Dense(40, activation="relu")(Location_time_input)
        out_location_time = Dense(20, kernel_regularizer=l2_reg)(out_location_time)

        out_dense = tf.Variable(2.0) * out_location_location + tf.Variable(2.0) * out_location_time
        out_dense = Dense(20, kernel_regularizer=l2_reg)(out_dense)

        omega_1 = (
            tf.Variable(1.0) * out_temporal
            + tf.Variable(1.0) * out_week_temporal
            + tf.Variable(1.0) * out_weekend_temporal
            + tf.Variable(1.0) * out_distance
            + tf.Variable(1.0) * out_duration
        )
        omega_1 = Dense(20, kernel_regularizer=l2_reg)(omega_1)
        omega_1 = tf.Variable(1.0) * out_dense + tf.Variable(1.0) * omega_1

        concat_ys_omega_1 = Concatenate()(
            [
                out_temporal,
                out_week_temporal,
                out_weekend_temporal,
                out_distance,
                out_duration,
                out_location_location,
                out_location_time,
                omega_1,
            ]
        )
        concat_ys_omega_1 = Dense(50)(concat_ys_omega_1)

        out_temporal2 = GATConv(
            20, kernel_regularizer=l2_reg, share_weights=self.share_weights, dropout_rate=self.dropout_skip
        )([Temporal_input, A_input])
        out_temporal2 = Dropout(self.dropout)(out_temporal2)
        out_temporal2 = GATConv(20, kernel_regularizer=l2_reg)([out_temporal2, A_input])

        out_week_temporal2 = GATConv(
            20, kernel_regularizer=l2_reg, share_weights=self.share_weights, dropout_rate=self.dropout_skip
        )([Temporal_week_input, A_week_input])
        out_week_temporal2 = Dropout(self.dropout)(out_week_temporal2)
        out_week_temporal2 = GATConv(20, kernel_regularizer=l2_reg)([out_week_temporal2, A_week_input])

        out_weekend_temporal2 = GATConv(
            20, kernel_regularizer=l2_reg, share_weights=self.share_weights, dropout_rate=self.dropout_skip
        )([Temporal_weekend_input, A_weekend_input])
        out_weekend_temporal2 = Dropout(self.dropout)(out_weekend_temporal2)
        out_weekend_temporal2 = GATConv(20, kernel_regularizer=l2_reg)([out_weekend_temporal2, A_weekend_input])

        out_distance2 = GATConv(
            20,
            kernel_regularizer=l2_reg,
        )([Distance_input, A_input])
        out_distance2 = Dropout(self.dropout)(out_distance2)
        out_distance2 = GATConv(20, kernel_regularizer=l2_reg)([out_distance2, A_input])

        out_duration2 = GATConv(
            20,
            kernel_regularizer=l2_reg,
        )([Duration_input, A_input])
        out_duration2 = Dropout(self.dropout)(out_duration2)
        out_duration2 = GATConv(20, kernel_regularizer=l2_reg)([out_duration2, A_input])

        out_location_location2 = GATConv(
            20,
            kernel_regularizer=l2_reg,
        )([Location_time_input, Location_location_input])
        out_location_location2 = Dropout(self.dropout)(out_location_location2)
        out_location_location2 = GATConv(20, kernel_regularizer=l2_reg)(
            [out_location_location2, Location_location_input]
        )
        out_location_time2 = Dense(40, activation="relu")(Location_time_input)
        out_location_time2 = Dense(20, kernel_regularizer=l2_reg)(out_location_time2)

        out_dense2 = tf.Variable(2.0) * out_location_location2 + tf.Variable(2.0) * out_location_time2
        out_dense2 = Dense(20, kernel_regularizer=l2_reg)(out_dense2)

        omega_2 = (
            tf.Variable(1.0) * out_temporal2
            + tf.Variable(1.0) * out_week_temporal2
            + tf.Variable(1.0) * out_weekend_temporal2
            + tf.Variable(1.0) * out_distance2
            + tf.Variable(1.0) * out_duration2
        )
        omega_2 = Dense(20, kernel_regularizer=l2_reg)(omega_2)
        omega_2 = tf.Variable(1.0) * out_dense2 + tf.Variable(1.0) * omega_2

        concat_ys_omega_2 = Concatenate()(
            [
                out_temporal2,
                out_week_temporal2,
                out_weekend_temporal2,
                out_distance2,
                out_duration2,
                out_location_location2,
                out_location_time2,
                omega_2,
            ]
        )
        concat_ys_omega_2 = Dense(50)(concat_ys_omega_2)

        if not self.baseline:
            # region ARMAConv
            out_embeddings = ARMAConv(
                20, kernel_regularizer=l2_reg, share_weights=self.share_weights, dropout_rate=self.dropout_skip
            )([User_embeddings_input, A_input])
            out_embeddings = Dropout(self.dropout)(out_embeddings)
            out_embeddings = ARMAConv(20, kernel_regularizer=l2_reg)([out_embeddings, A_input])

            out_embeddings2 = ARMAConv(
                20, kernel_regularizer=l2_reg, share_weights=self.share_weights, dropout_rate=self.dropout_skip
            )([User_embeddings_input, A_week_input])
            out_embeddings2 = Dropout(self.dropout)(out_embeddings2)
            out_embeddings2 = ARMAConv(20, kernel_regularizer=l2_reg)([out_embeddings2, A_week_input])

            out_embeddings3 = ARMAConv(
                20, kernel_regularizer=l2_reg, share_weights=self.share_weights, dropout_rate=self.dropout_skip
            )([User_embeddings_input, A_weekend_input])
            out_embeddings3 = Dropout(self.dropout)(out_embeddings3)
            out_embeddings3 = ARMAConv(20, kernel_regularizer=l2_reg)([out_embeddings3, A_weekend_input])

            omega_3 = (
                tf.Variable(1.0) * out_embeddings
                + tf.Variable(1.0) * out_embeddings2
                + tf.Variable(1.0) * out_embeddings3
            )
            omega_3 = Dense(20, kernel_regularizer=l2_reg)(omega_3)
            omega_3 = tf.Variable(1.0) * out_dense + tf.Variable(1.0) * omega_3

            concat_ys_omega_3 = Concatenate()(
                [
                    out_embeddings,
                    out_embeddings2,
                    out_embeddings3,
                    omega_3,
                ]
            )
            concat_ys_omega_3 = Dense(50)(concat_ys_omega_3)

            # region GATConv
            out_embeddings4 = GATConv(
                20, kernel_regularizer=l2_reg, share_weights=self.share_weights, dropout_rate=self.dropout_skip
            )([User_embeddings_input, A_input])
            out_embeddings4 = Dropout(self.dropout)(out_embeddings4)
            out_embeddings4 = GATConv(20, kernel_regularizer=l2_reg)([out_embeddings4, A_input])

            out_embeddings5 = GATConv(
                20, kernel_regularizer=l2_reg, share_weights=self.share_weights, dropout_rate=self.dropout_skip
            )([User_embeddings_input, A_week_input])
            out_embeddings5 = Dropout(self.dropout)(out_embeddings5)
            out_embeddings5 = GATConv(20, kernel_regularizer=l2_reg)([out_embeddings5, A_week_input])

            out_embeddings6 = GATConv(
                20, kernel_regularizer=l2_reg, share_weights=self.share_weights, dropout_rate=self.dropout_skip
            )([User_embeddings_input, A_weekend_input])
            out_embeddings6 = Dropout(self.dropout)(out_embeddings6)
            out_embeddings6 = GATConv(20, kernel_regularizer=l2_reg)([out_embeddings6, A_weekend_input])

            omega_4 = (
                tf.Variable(1.0) * out_embeddings4
                + tf.Variable(1.0) * out_embeddings5
                + tf.Variable(1.0) * out_embeddings6
            )
            omega_4 = Dense(20, kernel_regularizer=l2_reg)(omega_4)
            omega_4 = tf.Variable(1.0) * out_dense2 + tf.Variable(1.0) * omega_4

            concat_ys_omega_4 = Concatenate()(
                [
                    out_embeddings4,
                    out_embeddings5,
                    out_embeddings6,
                    omega_4,
                ]
            )
            concat_ys_omega_4 = Dense(50)(concat_ys_omega_4)

        c1 = (
            Concatenate()([concat_ys_omega_1, concat_ys_omega_2])
            if self.baseline
            else Concatenate()([concat_ys_omega_1, concat_ys_omega_2, concat_ys_omega_3, concat_ys_omega_4])
        )
        att = MultiHeadAttention(num_heads=2, key_dim=c1.shape[2])(c1, c1)
        out = Concatenate()([c1, att])
        out = Dense(50, activation="relu")(out)
        out = Dense(self.num_classes, activation="softmax")(out)

        if not self.baseline:
            inputs = [
                A_input,
                A_week_input,
                A_weekend_input,
                Temporal_input,
                Temporal_week_input,
                Temporal_weekend_input,
                Distance_input,
                Duration_input,
                Location_time_input,
                Location_location_input,
                User_embeddings_input,
            ]
        else:
            inputs = [
                A_input,
                A_week_input,
                A_weekend_input,
                Temporal_input,
                Temporal_week_input,
                Temporal_weekend_input,
                Distance_input,
                Duration_input,
                Location_time_input,
                Location_location_input,
            ]

        model = Model(inputs=inputs, outputs=[out])

        return model
