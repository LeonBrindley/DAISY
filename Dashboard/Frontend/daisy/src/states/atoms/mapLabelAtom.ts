import { atom } from 'recoil';
import { ImageSchemaOut } from '../../clients/general/index'; // Adjust the import path as necessary


export const selectetedMapLabelState = atom<String>({
  key: 'selectetedMapLabelState', // unique ID (with respect to other atoms/selectors)
  default: "grass", // default value (aka initial value)
});
