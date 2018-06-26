   def proper_yolo_nms(self, y_sing_pred):
        # NMS need to be applied per class, since two different boxes could predict with high confidence
        # two objects that have high IOU
        # At the same time, even though NMS has to be done per class, it can only be done with max values
        # of P(O) * P(Class|O) since we want to avoid same box predicting 2 overlapping objects.
        # Doing both these things turns out to be a fucking pain.

        # CONSIDER USING tf.while_loop for the FOR

        b_xy = tf.sigmoid(y_sing_pred[..., 0:2]) + YoloParams.c_grid[0]
        b_wh = tf.exp(y_sing_pred[..., 2:4])*YoloParams.anchors[0]
        b_xy1 = b_xy - b_wh / 2.
        b_xy2 = b_xy + b_wh / 2.
        boxes = tf.concat([b_xy1, b_xy2], axis=-1)

        
        scores_all = tf.expand_dims(tf.sigmoid(y_sing_pred[..., 4]), axis=-1) * tf.nn.softmax(y_sing_pred[...,5:])
        indicator_detection = scores_all > self.detection_threshold

        scores_all = scores_all * tf.to_float(indicator_detection)

        classes = tf.argmax(scores_all, axis=-1)

        scores = tf.reduce_max(scores_all, axis=-1)
        
        flatten_boxes = tf.reshape(boxes, 
            shape=(YoloParams.GRID_SIZE*YoloParams.GRID_SIZE*YoloParams.NUM_BOUNDING_BOXES, 4))
        flatten_scores = tf.reshape(scores, 
            shape=(YoloParams.GRID_SIZE*YoloParams.GRID_SIZE*YoloParams.NUM_BOUNDING_BOXES, ))
        flatten_classes = tf.reshape(classes, 
            shape=(YoloParams.GRID_SIZE*YoloParams.GRID_SIZE*YoloParams.NUM_BOUNDING_BOXES, ))

        output_boxes = []
        output_scores = []
        output_classes = []
        for c in range(YoloParams.NUM_CLASSES):
            if tf.reduce_sum(tf.to_float(tf.equal(flatten_classes, c))) > 0:
                filtered_flatten_boxes = tf.boolean_mask(flatten_boxes, tf.equal(flatten_classes, c))
                filtered_flatten_scores = tf.boolean_mask(flatten_scores, tf.equal(flatten_classes, c))
                filtered_flatten_classes = tf.boolean_mask(flatten_classes, tf.equal(flatten_classes, c))

                selected_indices = tf.image.non_max_suppression( 
                    filtered_flatten_boxes, filtered_flatten_scores, self.max_boxes, self.iou_threshold)

                selected_boxes = K.gather(filtered_flatten_boxes, selected_indices)
                selected_scores = K.gather(filtered_flatten_scores, selected_indices)
                selected_classes = K.gather(filtered_flatten_classes, selected_indices)


                output_boxes.append( selected_boxes )
                output_scores.append( selected_scores )
                output_classes.append( selected_classes )


        print(output_boxes)

        print(tf.concat(output_boxes, axis=-1).eval()) 
        print(tf.concat(output_scores, axis=-1).eval())
        print(tf.concat(output_classes, axis=-1).eval())

        return tf.concat(output_boxes, axis=-1), tf.concat(output_scores, axis=-1), tf.concat(output_classes, axis=-1)

