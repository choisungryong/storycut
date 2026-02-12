[DerivedCutController 보완 지시 – 클로즈업 목/배 샷 문제 해결]

증상: closeup에서 목이 잘리거나 배만 나오는 컷이 반복됨. 기존보다 나아졌지만 QA가 아직 불충분.

필수 수정(이번 커밋에서 반영):
1) CLOSEUP/MEDIUM은 face_bbox 필수.
   - face_bbox 없으면 해당 shot_type은 FAIL 처리하고 파라미터 재시도.
   - 재시도 N회 실패 시 CLOSEUP 금지하고 MEDIUM/WIDE로 강등.

2) QA에 '프레이밍 품질' 조건을 추가:
   - eyes_y(대략): face_top + 0.35*face_h
     조건: 0.22H <= eyes_y <= 0.42H (벗어나면 FAIL)
   - chin_y(대략): face_top + 0.90*face_h
     조건: (frame_bottom - chin_y) >= 0.12H (벗어나면 FAIL)
   - start/end 프레임 모두 검사(kenburns가 있으면 end에서도 동일 QA 적용)

3) CLOSEUP 스케일 범위를 더 보수적으로:
   - CLOSEUP: 1.10 ~ 1.18 (상한 1.25는 사실상 금지 수준으로 낮춤)
   - MEDIUM: 1.05 ~ 1.12
   - WIDE: 1.00 ~ 1.05

4) 다인(얼굴 2개 이상) 케이스:
   - 가장 큰 얼굴 또는 프레임 중앙에 가장 가까운 얼굴을 dominant face로 선택.
   - dominant face가 불명확하면 CLOSEUP 금지하고 MEDIUM으로 강등.

로그 강화:
- bbox_source(face/person/saliency), shot_type, scale_from/to
- eyes_y_ratio, chin_margin_ratio
- qa_fail_reason(폐기 컷 포함)
