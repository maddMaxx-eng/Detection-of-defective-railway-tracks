/**
 * Held-out TEST-set results for the two trained checkpoints, as reported by the project owner.
 * The evaluation artifacts (artifacts/metrics.json) are not stored in the repository, so these
 * figures were not re-computed when the web app was built. Update this file if you re-train.
 * (Validation figures stored inside each checkpoint are shown separately and are read directly
 * from public/models/manifest.json.)
 */
export const REPORTED_TEST_METRICS: Record<string, { accuracy: number; macroF1: number }> = {
  efficientnet_b0: { accuracy: 0.9364, macroF1: 0.8836 },
  mobilenet_v2: { accuracy: 0.9227, macroF1: 0.8572 },
};

/** Preferred model according to the reported test results (higher macro-F1). */
export const PREFERRED_MODEL = "efficientnet_b0";
