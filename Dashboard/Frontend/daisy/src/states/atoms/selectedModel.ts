import { atom } from 'recoil';

export const selectedModelState = atom<string>({
  key: 'selectedModelState', // unique ID (with respect to other atoms/selectors)
  default: "sensor-cdt-daisy-models/inaturalist_uf5_pFalse_dp0.1_wd1e-05_lr0.0001.zip", // default value (aka initial value)
});
