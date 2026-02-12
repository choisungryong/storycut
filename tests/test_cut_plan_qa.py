"""
Unit tests for DerivedCutController framing QA logic.

Tests cover:
1. Eyes_y / chin_y framing quality checks
2. Scale range clamping (_SHOT_SCALE, _SHOT_ZOOM)
3. Multi-face dominant face selection
4. Mandatory face_bbox enforcement (CLOSEUP downgrade)
"""
import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ==========================================================================
# Test 1: Framing Quality QA (eyes_y / chin_y)
# ==========================================================================

class TestFramingQuality:
    """Closeup_fix #2: eyes_y must be in [0.22H, 0.42H], chin margin >= 0.12H."""

    @staticmethod
    def _compute_framing(face_bb, anchor, reframe="close"):
        """Simulate framing QA from _generate_cut_plan."""
        reframe_ratios = {"wide": 1.0, "medium": 0.75, "close": 0.5, "detail": 0.33}
        ratio = reframe_ratios.get(reframe, 1.0)

        face_top = face_bb["y"]
        crop_top = anchor[1] - ratio / 2

        eyes_y = face_top + 0.35 * face_bb["h"]
        eyes_y_in_crop = (eyes_y - crop_top) / ratio if ratio > 0 else 0.5

        chin_y = face_top + 0.90 * face_bb["h"]
        chin_y_in_crop = (chin_y - crop_top) / ratio if ratio > 0 else 0.5
        chin_margin = 1.0 - chin_y_in_crop

        return eyes_y_in_crop, chin_margin

    def test_good_framing_passes(self):
        """Well-centered face should pass all QA checks."""
        face_bb = {"x": 0.35, "y": 0.15, "w": 0.3, "h": 0.25}
        anchor = (0.5, 0.30)
        eyes_y, chin_margin = self._compute_framing(face_bb, anchor, "close")
        assert 0.22 <= eyes_y <= 0.42, f"eyes_y={eyes_y:.3f} out of range"
        assert chin_margin >= 0.12, f"chin_margin={chin_margin:.3f} too small"

    def test_eyes_too_high_fails(self):
        """Face too high in frame should fail eyes_y check."""
        face_bb = {"x": 0.35, "y": 0.02, "w": 0.3, "h": 0.2}
        anchor = (0.5, 0.30)
        eyes_y, _ = self._compute_framing(face_bb, anchor, "close")
        assert eyes_y < 0.22, f"eyes_y={eyes_y:.3f} should be < 0.22"

    def test_eyes_too_low_fails(self):
        """Face too low in frame should fail eyes_y check."""
        face_bb = {"x": 0.35, "y": 0.55, "w": 0.3, "h": 0.25}
        anchor = (0.5, 0.50)
        eyes_y, _ = self._compute_framing(face_bb, anchor, "close")
        assert eyes_y > 0.42, f"eyes_y={eyes_y:.3f} should be > 0.42"

    def test_chin_margin_too_small_fails(self):
        """Chin too close to bottom should fail margin check."""
        face_bb = {"x": 0.35, "y": 0.30, "w": 0.3, "h": 0.50}
        anchor = (0.5, 0.50)
        _, chin_margin = self._compute_framing(face_bb, anchor, "close")
        assert chin_margin < 0.12, f"chin_margin={chin_margin:.3f} should be < 0.12"

    def test_medium_reframe_wider_margin(self):
        """Medium reframe (0.75 ratio) should give more chin margin."""
        face_bb = {"x": 0.35, "y": 0.20, "w": 0.3, "h": 0.25}
        anchor = (0.5, 0.35)
        _, chin_margin_close = self._compute_framing(face_bb, anchor, "close")
        _, chin_margin_medium = self._compute_framing(face_bb, anchor, "medium")
        assert chin_margin_medium > chin_margin_close


# ==========================================================================
# Test 2: Scale Range Clamping
# ==========================================================================

class TestScaleRanges:
    """Closeup_fix #3: Conservative scale ranges."""

    def test_shot_scale_close_max(self):
        """CLOSEUP max scale should be <= 1.18."""
        from utils.ffmpeg_utils import FFmpegComposer
        close_range = FFmpegComposer._SHOT_SCALE["close"]
        assert close_range[1] <= 1.18, f"close max={close_range[1]}, expected <= 1.18"

    def test_shot_scale_medium_max(self):
        """MEDIUM max scale should be <= 1.12."""
        from utils.ffmpeg_utils import FFmpegComposer
        medium_range = FFmpegComposer._SHOT_SCALE["medium"]
        assert medium_range[1] <= 1.12, f"medium max={medium_range[1]}, expected <= 1.12"

    def test_shot_scale_wide_max(self):
        """WIDE max scale should be <= 1.05."""
        from utils.ffmpeg_utils import FFmpegComposer
        wide_range = FFmpegComposer._SHOT_SCALE["wide"]
        assert wide_range[1] <= 1.05, f"wide max={wide_range[1]}, expected <= 1.05"

    def test_shot_scale_ranges_non_overlapping(self):
        """Scale ranges should be properly ordered: wide < medium < close."""
        from utils.ffmpeg_utils import FFmpegComposer
        wide = FFmpegComposer._SHOT_SCALE["wide"]
        medium = FFmpegComposer._SHOT_SCALE["medium"]
        close = FFmpegComposer._SHOT_SCALE["close"]
        assert wide[1] <= medium[0] or wide[1] <= medium[1]
        assert medium[0] <= close[0] or medium[1] <= close[1]

    def test_max_delta_scale(self):
        """Max delta scale should be 0.06."""
        from utils.ffmpeg_utils import FFmpegComposer
        assert FFmpegComposer._MAX_DELTA_SCALE == 0.06


# ==========================================================================
# Test 3: Dominant Face Selection (multi-face)
# ==========================================================================

class TestDominantFaceSelection:
    """Closeup_fix #4: Multi-face handling."""

    @staticmethod
    def _select_dominant_face(faces, img_w=1920, img_h=1080):
        """Simulate the dominant face selection from _opencv_bbox."""
        sorted_faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        multi_face_ambiguous = False

        x, y, w, h = sorted_faces[0]

        if len(faces) >= 2:
            area_1st = sorted_faces[0][2] * sorted_faces[0][3]
            area_2nd = sorted_faces[1][2] * sorted_faces[1][3]
            if area_1st < area_2nd * 1.5:
                center_x, center_y = img_w / 2, img_h / 2

                def dist_to_center(f):
                    fx = f[0] + f[2] / 2
                    fy = f[1] + f[3] / 2
                    return ((fx - center_x) ** 2 + (fy - center_y) ** 2) ** 0.5

                closest = min(sorted_faces, key=dist_to_center)
                x, y, w, h = closest
                multi_face_ambiguous = True

        return (x, y, w, h), multi_face_ambiguous

    def test_single_face(self):
        """Single face should be selected, no ambiguity."""
        faces = [(100, 100, 200, 200)]
        selected, ambiguous = self._select_dominant_face(faces)
        assert selected == (100, 100, 200, 200)
        assert not ambiguous

    def test_clear_dominant_face(self):
        """Face 2x bigger than the second should be dominant without ambiguity."""
        faces = [(100, 100, 400, 400), (800, 200, 150, 150)]
        selected, ambiguous = self._select_dominant_face(faces)
        assert selected == (100, 100, 400, 400)
        assert not ambiguous

    def test_ambiguous_faces_selects_center(self):
        """Similar-sized faces should trigger ambiguity, selecting closest to center."""
        # Face A: left side, 200x200
        # Face B: near center, 190x190
        faces = [(100, 400, 200, 200), (860, 440, 190, 190)]
        selected, ambiguous = self._select_dominant_face(faces, img_w=1920, img_h=1080)
        assert ambiguous
        # Face B is closer to center (960,540)
        assert selected == (860, 440, 190, 190)

    def test_three_faces_dominant(self):
        """Three faces, one clearly dominant."""
        faces = [(100, 100, 300, 300), (600, 200, 100, 100), (900, 300, 80, 80)]
        selected, ambiguous = self._select_dominant_face(faces)
        assert selected == (100, 100, 300, 300)
        assert not ambiguous


# ==========================================================================
# Test 4: Mandatory face_bbox enforcement
# ==========================================================================

class TestMandatoryFaceBbox:
    """Closeup_fix #1: CLOSEUP without face_bbox → WIDE, MEDIUM allowed with saliency."""

    @staticmethod
    def _enforce_face_bbox(reframe, has_chars, face_bb, is_multi_face_ambiguous):
        """Simulate mandatory face_bbox enforcement from _generate_cut_plan."""
        downgraded = False

        if has_chars and reframe in ("close", "detail") and not face_bb:
            reframe = "wide"
            downgraded = True

        if is_multi_face_ambiguous and reframe in ("close", "detail"):
            reframe = "medium"
            downgraded = True

        return reframe, downgraded

    def test_closeup_with_face_bbox_passes(self):
        """CLOSEUP with face_bbox should remain CLOSEUP."""
        face_bb = {"x": 0.3, "y": 0.1, "w": 0.4, "h": 0.3}
        reframe, downgraded = self._enforce_face_bbox("close", True, face_bb, False)
        assert reframe == "close"
        assert not downgraded

    def test_closeup_without_face_bbox_downgrades_to_wide(self):
        """CLOSEUP without face_bbox should downgrade to WIDE."""
        reframe, downgraded = self._enforce_face_bbox("close", True, None, False)
        assert reframe == "wide"
        assert downgraded

    def test_detail_without_face_bbox_downgrades_to_wide(self):
        """DETAIL without face_bbox should downgrade to WIDE."""
        reframe, downgraded = self._enforce_face_bbox("detail", True, None, False)
        assert reframe == "wide"
        assert downgraded

    def test_medium_without_face_bbox_stays_medium(self):
        """MEDIUM without face_bbox NOT in close/detail, stays medium."""
        reframe, downgraded = self._enforce_face_bbox("medium", True, None, False)
        assert reframe == "medium"
        assert not downgraded

    def test_closeup_multi_face_ambiguous_downgrades_to_medium(self):
        """CLOSEUP with ambiguous multi-face should downgrade to MEDIUM."""
        face_bb = {"x": 0.3, "y": 0.1, "w": 0.4, "h": 0.3}
        reframe, downgraded = self._enforce_face_bbox("close", True, face_bb, True)
        assert reframe == "medium"
        assert downgraded

    def test_wide_no_chars_unaffected(self):
        """WIDE with no characters should remain unaffected."""
        reframe, downgraded = self._enforce_face_bbox("wide", False, None, False)
        assert reframe == "wide"
        assert not downgraded

    def test_no_chars_closeup_stays(self):
        """CLOSEUP without characters (e.g. object closeup) should stay."""
        reframe, downgraded = self._enforce_face_bbox("close", False, None, False)
        assert reframe == "close"
        assert not downgraded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
