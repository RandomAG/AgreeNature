"""
PEWTAF Raspberry Pi edge orchestration.

CPU:
    Frame acquisition
    Preprocessing
    TTVM
    TAN
    Metadata generation
    Communication
    EFPM

Hailo-8L:
    Neural inference
"""

class RaspberryPiPipeline:

    def __init__(
        self,
        preprocessor,
        hailo_backend,
        postprocessor,
        temporal_validator,
        threat_assessor,
        metadata_generator,
        frame_processor,
        publisher=None
    ):
        self.preprocessor = preprocessor
        self.hailo = hailo_backend
        self.postprocessor = postprocessor
        self.temporal_validator = temporal_validator
        self.threat_assessor = threat_assessor
        self.metadata_generator = metadata_generator
        self.frame_processor = frame_processor
        self.publisher = publisher

    def process_frame(self, frame):
        """
        Execute one complete frame through the PEWTAF pipeline.
        """

        self.frame_processor.store(frame)

        try:
            # CPU: preprocessing
            input_tensor = self.preprocessor.process(frame)

            # NPU: Hybrid YOLOv10-MobileViTv3 inference
            raw_output = self.hailo.infer(input_tensor)

            # CPU: detector output decoding
            detections = self.postprocessor.process(raw_output)

            # CPU: temporal validation
            validated = self.temporal_validator.update(detections)

            if validated is None:
                return None

            # CPU: threat assessment
            assessment = self.threat_assessor.evaluate(
                bbox=validated.bbox,
                weapon_class=validated.class_name,
                confidence=validated.confidence,
                persistence=validated.persistence
            )

            # CPU: metadata generation
            metadata = self.metadata_generator.generate(
                assessment
            )


            # Communication: metadata only
            if self.publisher is not None:
                self.publisher.publish(metadata)

            return metadata

        finally:

            # EFPM: immediately release frame
            self.frame_processor.discard()