// src/state/atoms/inferenceResultsAtom.ts
import { atom } from 'recoil';
import { InferenceResultSchemaOut } from '../../clients/inference/index'; // Adjust the import path as necessary


export const inferenceResultsState = atom<InferenceResultSchemaOut[]>({
  key: 'inferenceResultsState', // unique ID (with respect to other atoms/selectors)
  default: [], // default value (aka initial value)
});
