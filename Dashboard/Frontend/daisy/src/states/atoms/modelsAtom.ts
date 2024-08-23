// src/state/atoms/modelsAtom.ts
import { atom } from 'recoil';
import { VisionModelSchemaOut } from '../../clients/inference/index'; // Adjust the import path as necessary


export const modelsState = atom<VisionModelSchemaOut[]>({
  key: 'modelsState', // unique ID (with respect to other atoms/selectors)
  default: [], // default value (aka initial value)
});
