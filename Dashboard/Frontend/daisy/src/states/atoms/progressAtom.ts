// src/state/atoms/imagesAtom.ts
import { atom } from 'recoil';
import { InferenceProgressSchemaOut } from '../../clients/general/index'; // Adjust the import path as necessary


export const inferenceProgressState = atom<InferenceProgressSchemaOut>({
  key: 'inferenceProgressState', // unique ID (with respect to other atoms/selectors)
  default: {
    currently_training: false,
    percentage_completed: 100,
  },
});
