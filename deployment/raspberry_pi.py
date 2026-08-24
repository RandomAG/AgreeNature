"""
Raspberry Pi orchestration layer for PEWTAF.

The Raspberry Pi performs:
    - video acquisition
    - preprocessing
    - temporal validation
    - threat assessment
    - metadata generation
    - communication

Neural inference is delegated to the Hailo-8L backend.
"""

class RaspberryPiPipeline:
    """Coordinate the edge-side PEWTAF processing pipeline."""

    def __init__(
        self,
        preprocessor,
        detector,
        temporal_validator,
        threat_assessor,
        metadata_generator,
        frame_processor,
        publisher=None
    ):
        self.preprocessor = preprocessor
        self.detector = detector
        self.temporal_validator = temporal_validator
        self.threat_assessor = threat_assessor
        self.metadata_generator = metadata_generator
        self.frame_processor = frame_processor
        self.publisher = publisher

    def process_frame(self, frame):
        """
        Process one surveillance frame through the complete
        edge-side pipeline.
        """

        try:
            # 1. Temporarily retain the frame.
            self.frame_processor.store(frame)

            # 2. Raspberry Pi preprocessing.
            input_tensor = self.preprocessor.process(frame)

            # 3. Hailo-8L neural inference.
            raw_predictions = self.detector.infer(
                input_tensor
            )

            # 4. Convert model output to detections.
            detections = self.detector.postprocess(
                raw_predictions
            )

            # 5. Temporal threat validation.
            validated = self.temporal_validator.update(
                detections
            )

            if validated is None:
                return None

            # 6. Threat assessment on Raspberry Pi CPU.
            assessment = self.threat_assessor.evaluate(
                bbox=validated.bbox,
                weapon_class=validated.class_name,
                confidence=validated.confidence,
                persistence=validated.persistence
            )

            # 7. Generate compact metadata.
            metadata = self.metadata_generator.generate(
                assessment
            )

            # 8. Transmit metadata only.
            if self.publisher is not None:
                self.publisher.publish(metadata)

            return metadata

        finally:
            # 9. EFPM: release the temporary frame.
            self.frame_processor.discard()