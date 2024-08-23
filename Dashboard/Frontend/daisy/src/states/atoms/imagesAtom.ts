// src/state/atoms/imagesAtom.ts
import { atom } from 'recoil';
import { ImageSchemaOut } from '../../clients/general/index'; // Adjust the import path as necessary


export const imagesState = atom<ImageSchemaOut[]>({
  key: 'imagesState', // unique ID (with respect to other atoms/selectors)
  default: [], // default value (aka initial value)
});
