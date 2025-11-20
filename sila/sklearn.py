"""
scikit-learn 스타일의 SILA 래퍼.

SILATransformer 는 기존 함수형 API(sila, sila_estimate)를 감싼 뒤
fit/transform 인터페이스를 제공해 추론 파이프라인에 통합하기 쉽게 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import warnings

from .estimate import sila_estimate
from .sila import sila


def _coerce_array(values, name: str) -> np.ndarray:
    """입력 값을 numpy array 로 통일."""
    try:
        return np.asarray(values)
    except Exception as exc:  # pragma: no cover - numpy 예외 메시지 전달
        raise TypeError(f"{name}을(를) 배열로 변환할 수 없습니다.") from exc


def _clean_dataframe(
    data: pd.DataFrame,
    age_col: str,
    value_col: str,
    subid_col: str,
    *,
    invalid_policy: str,
    stage: str,
    require_longitudinal: bool,
    track_index: bool = False,
    allow_missing_values: bool = False,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Validate and sanitize DataFrame inputs according to policy."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data는 pandas.DataFrame 이어야 합니다.")

    missing = [col for col in (age_col, value_col, subid_col) if col not in data.columns]
    if missing:
        raise KeyError(f"DataFrame에 다음 컬럼이 없습니다: {', '.join(missing)}")

    if invalid_policy not in {"drop", "raise"}:
        raise ValueError("invalid_policy는 'drop' 또는 'raise'만 지원합니다.")

    sanitized = data.copy()
    kept_positions: Optional[np.ndarray] = None
    if track_index:
        sanitized["_orig_pos"] = np.arange(len(sanitized), dtype=np.int64)

    age_numeric = pd.to_numeric(sanitized[age_col], errors="coerce")
    value_numeric = pd.to_numeric(sanitized[value_col], errors="coerce")
    subid_values = sanitized[subid_col]

    age_finite = age_numeric.notna() & np.isfinite(age_numeric)
    value_finite = value_numeric.notna() & np.isfinite(value_numeric)
    subid_valid = subid_values.notna()

    if allow_missing_values:
        finite_mask = age_finite & subid_valid
    else:
        finite_mask = age_finite & value_finite & subid_valid

    invalid_count = len(sanitized) - finite_mask.sum()
    if invalid_count:
        message = (
            f"[{stage}] {invalid_count}행에서 age/value 비유한 값 또는 subid 결측이 발견되었습니다."
        )
        if invalid_policy == "drop":
            warnings.warn(message + " 해당 행을 제외합니다.", UserWarning, stacklevel=3)
            sanitized = sanitized.loc[finite_mask].copy()
            age_numeric = age_numeric.loc[finite_mask]
            value_numeric = value_numeric.loc[finite_mask]
        else:
            raise ValueError(message)

    if sanitized.empty:
        raise ValueError(f"[{stage}] 유효한 데이터가 없습니다.")

    sanitized[age_col] = age_numeric.astype(np.float64).values
    sanitized[value_col] = value_numeric.astype(np.float64).values

    # Sort by subject then age for deterministic processing
    sanitized = sanitized.sort_values(by=[subid_col, age_col]).reset_index(drop=True)

    if require_longitudinal:
        counts = sanitized[subid_col].value_counts()
        valid_ids = counts[counts >= 2].index
        filtered = sanitized[sanitized[subid_col].isin(valid_ids)].copy()
        removed_rows = len(sanitized) - len(filtered)
        removed_subjects = (counts < 2).sum()
        if removed_rows:
            warnings.warn(
                f"[{stage}] {removed_subjects}명의 피험자에서 관측치가 1개뿐이라 제외했습니다.",
                UserWarning,
                stacklevel=3,
            )
        sanitized = filtered.reset_index(drop=True)

    if sanitized.empty:
        raise ValueError(f"[{stage}] 필터링 후 남은 데이터가 없습니다.")

    if track_index:
        kept_positions = sanitized["_orig_pos"].to_numpy(dtype=np.int64)
        sanitized = sanitized.drop(columns=["_orig_pos"])

    return sanitized, kept_positions


def _prepare_inputs(
    data: pd.DataFrame,
    age_col: str,
    value_col: str,
    subid_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pandas DataFrame과 컬럼명을 받아 numpy 배열로 통합한다.
    """
    age = data[age_col]
    value = data[value_col]
    subid = data[subid_col]

    age_arr = _coerce_array(age, age_col)
    value_arr = _coerce_array(value, value_col)
    subid_arr = _coerce_array(subid, subid_col)

    if not (len(age_arr) == len(value_arr) == len(subid_arr)):
        raise ValueError("age/value/subid 길이가 일치하지 않습니다.")

    return (
        np.asarray(age_arr, dtype=np.float64),
        np.asarray(value_arr, dtype=np.float64),
        np.asarray(subid_arr),
    )


@dataclass
class _FitSummary:
    """fit 결과에 대한 메타데이터 요약."""

    n_subjects: int
    n_observations: int
    smoothing_kernel: float
    age_column: str
    value_column: str
    subid_column: str


class SILATransformer:
    """
    scikit-learn 스타일로 SILA를 감싼 변환기.

    Parameters
    ----------
    dt : float
        ILLA 적분 간격(년). 기본값 0.25.
    val0 : float
        threshold 값. 기본값 0.79.
    maxi : int
        통합 반복 횟수 제한. 기본 200.
    sk : array-like or None
        smoothing kernel 후보. None이면 기본 검색(0~0.5, step 0.05).
    align_event : str
        transform 시 sila_estimate에 전달할 alignment 기준 (first/last/all).
    extrap_years : float
        transform 시 extrapolation 연수.
    truncate_aget0 : str
        transform 시 threshold age truncation 여부.
    """

    def __init__(
        self,
        dt: float = 0.25,
        val0: float = 0.79,
        maxi: int = 200,
        sk: Optional[Sequence[float]] = None,
        *,
        align_event: str = "last",
        extrap_years: float = 3.0,
        truncate_aget0: str = "yes",
        invalid_policy: str = "drop",
    ) -> None:
        self.dt = dt
        self.val0 = val0
        self.maxi = maxi
        self.sk = sk
        self.align_event = align_event
        self.extrap_years = extrap_years
        self.truncate_aget0 = truncate_aget0
        self.invalid_policy = invalid_policy

        self.tsila_: Optional[pd.DataFrame] = None
        self.tdrs_: Optional[pd.DataFrame] = None
        self.is_fitted_: bool = False
        self.age_column_: Optional[str] = None
        self.value_column_: Optional[str] = None
        self.subid_column_: Optional[str] = None
        self.last_estimates_: Optional[pd.DataFrame] = None
        self.summary_: Optional[_FitSummary] = None

    # ------------------------------------------------------------------
    # scikit-learn 호환 유틸
    def get_params(self, deep: bool = True):
        return {
            "dt": self.dt,
            "val0": self.val0,
            "maxi": self.maxi,
            "sk": self.sk,
            "align_event": self.align_event,
            "extrap_years": self.extrap_years,
            "truncate_aget0": self.truncate_aget0,
            "invalid_policy": self.invalid_policy,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"알 수 없는 파라미터: {key}")
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    def fit(
        self,
        data: pd.DataFrame,
        *,
        age: str = "age",
        value: str = "value",
        subid: str = "subid",
    ):
        """
        학습 데이터를 이용해 SILA 곡선을 적합한다.

        Parameters
        ----------
        data : pandas.DataFrame
            'age', 'value', 'subid' 컬럼을 포함하는 입력.
        age/value/subid : str, optional
            사용할 컬럼명. 기본값은 각각 'age', 'value', 'subid'.

        Returns
        -------
        self : SILATransformer
        """
        clean_df, _ = _clean_dataframe(
            data,
            age,
            value,
            subid,
            invalid_policy=self.invalid_policy,
            stage="fit",
            require_longitudinal=True,
            track_index=False,
        )

        age_arr, val_arr, subid_arr = _prepare_inputs(clean_df, age, value, subid)

        tsila, tdrs = sila(
            age_arr,
            val_arr,
            subid_arr,
            dt=self.dt,
            val0=self.val0,
            maxi=self.maxi,
            sk=self.sk,
        )

        self.tsila_ = tsila
        self.tdrs_ = tdrs
        self.is_fitted_ = True
        self.age_column_ = age
        self.value_column_ = value
        self.subid_column_ = subid
        self.summary_ = _FitSummary(
            n_subjects=len(np.unique(subid_arr)),
            n_observations=len(age_arr),
            smoothing_kernel=float(tdrs["skern"].iloc[0]) if "skern" in tdrs else float("nan"),
            age_column=age,
            value_column=value,
            subid_column=subid,
        )
        return self

    def transform(
        self,
        data: pd.DataFrame,
        *,
        align_event: Optional[str] = None,
        extrap_years: Optional[float] = None,
        truncate_aget0: Optional[str] = None,
    ) -> np.ndarray:
        """
        적합된 SILA 곡선을 사용해 새로운 관측치의 추정 시간(estdtt0)을 반환한다.

        Parameters
        ----------
        data : pandas.DataFrame
            fit 단계와 동일한 컬럼명을 가진 입력
        align_event / extrap_years / truncate_aget0 :
            transform 시점에서 override 가능

        Returns
        -------
        np.ndarray
            입력 DataFrame 순서와 동일한 길이의 estdtt0 배열
        """
        self._check_is_fitted()
        clean_df, positions = _clean_dataframe(
            data,
            self.age_column_,
            self.value_column_,
            self.subid_column_,
            invalid_policy=self.invalid_policy,
            stage="transform",
            require_longitudinal=False,
            track_index=True,
            allow_missing_values=True,
        )

        value_series = clean_df[self.value_column_]
        has_value = value_series.notna() & np.isfinite(value_series)

        if not has_value.any():
            raise ValueError("[transform] 유효한 측정값이 없어 추론할 수 없습니다.")

        measurement_df = clean_df.loc[has_value].copy()
        age_arr, val_arr, subid_arr = _prepare_inputs(
            measurement_df,
            self.age_column_,
            self.value_column_,
            self.subid_column_,
        )

        estimates = sila_estimate(
            self.tsila_,
            age_arr,
            val_arr,
            subid_arr,
            align_event=align_event or self.align_event,
            extrap_years=extrap_years if extrap_years is not None else self.extrap_years,
            truncate_aget0=truncate_aget0 or self.truncate_aget0,
        )
        self.last_estimates_ = estimates.copy()

        estdtt0_sorted = np.full(len(clean_df), np.nan, dtype=np.float64)
        estdtt0_sorted[measurement_df.index.to_numpy()] = estimates["estdtt0"].to_numpy(
            dtype=np.float64, copy=True
        )

        infer_mask = ~has_value
        if infer_mask.any():
            subject_estaget0 = estimates.groupby("subid")["estaget0"].median()
            mapped = clean_df.loc[infer_mask, self.subid_column_].map(subject_estaget0)
            missing_subjects = mapped.isna()
            if missing_subjects.any():
                missing_count = missing_subjects.sum()
                warnings.warn(
                    f"[transform] 값이 없는 {missing_count}개 행은 추정에 사용된 측정이 없어 NaN으로 남습니다.",
                    UserWarning,
                    stacklevel=3,
                )
            inferred = (
                clean_df.loc[infer_mask, self.age_column_].astype(np.float64) - mapped.astype(np.float64)
            )
            estdtt0_sorted[infer_mask.to_numpy()] = inferred.to_numpy()

        if positions is None:
            raise RuntimeError("원본 행 위치 정보가 없습니다.")

        output = np.full(len(data), np.nan, dtype=np.float64)
        output[positions] = estdtt0_sorted
        return output

    def fit_transform(
        self,
        data: pd.DataFrame,
        *,
        age: str = "age",
        value: str = "value",
        subid: str = "subid",
        align_event: Optional[str] = None,
        extrap_years: Optional[float] = None,
        truncate_aget0: Optional[str] = None,
    ) -> np.ndarray:
        """편의 메서드: fit 후 estdtt0 배열을 반환."""
        self.fit(data=data, age=age, value=value, subid=subid)
        return self.transform(
            data=data,
            align_event=align_event,
            extrap_years=extrap_years,
            truncate_aget0=truncate_aget0,
        )

    # ------------------------------------------------------------------
    def _check_is_fitted(self):
        if (
            not self.is_fitted_
            or self.tsila_ is None
            or self.age_column_ is None
            or self.value_column_ is None
            or self.subid_column_ is None
        ):
            raise RuntimeError("SILATransformer가 아직 fit 되지 않았습니다.")
