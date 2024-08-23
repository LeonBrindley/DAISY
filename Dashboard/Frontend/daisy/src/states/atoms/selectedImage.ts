
import { atom } from 'recoil';


export const selectedImageIdState = atom<string | null>({
    key: 'selectedImageIdState',
    default: null,
  });

