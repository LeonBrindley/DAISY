
import { atom } from 'recoil';


export const selectedFieldState = atom<string | null>({
    key: 'selectedFieldState',
    default: null,
  });

