/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Body_upload_file_upload_post } from '../models/Body_upload_file_upload_post';
import type { ImageSchemaIn } from '../models/ImageSchemaIn';
import type { ImageSchemaOut } from '../models/ImageSchemaOut';
import type { InferenceProgressSchemaOut } from '../models/InferenceProgressSchemaOut';
import type { ProductSchemaIn } from '../models/ProductSchemaIn';
import type { ProductSchemaOut } from '../models/ProductSchemaOut';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class DefaultService {
    /**
     * Upload File
     * @param formData
     * @returns any Successful Response
     * @throws ApiError
     */
    public static uploadFileUploadPost(
        formData: Body_upload_file_upload_post,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/upload',
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Image File
     * @param imageId
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getImageFileGetImageFileImageIdGet(
        imageId: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/get_image_file/{image_id}',
            path: {
                'image_id': imageId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Prediction Image File
     * @param predictionId
     * @param label
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getPredictionImageFileGetPredictionImageFilePredictionIdGet(
        predictionId: string,
        label: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/get_prediction_image_file/{prediction_id}',
            path: {
                'prediction_id': predictionId,
            },
            query: {
                'label': label,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Image
     * @param requestBody
     * @returns ImageSchemaOut Successful Response
     * @throws ApiError
     */
    public static createImageInsertImgPost(
        requestBody: ImageSchemaIn,
    ): CancelablePromise<ImageSchemaOut> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/insert_img',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Image
     * @param imageId
     * @returns ImageSchemaOut Successful Response
     * @throws ApiError
     */
    public static createImageGetImgImageIdGet(
        imageId: string,
    ): CancelablePromise<ImageSchemaOut> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/get_img/{image_id}',
            path: {
                'image_id': imageId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get All Images
     * @returns ImageSchemaOut Successful Response
     * @throws ApiError
     */
    public static getAllImagesGetAllImagesGet(): CancelablePromise<Array<ImageSchemaOut>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/get_all_images',
        });
    }
    /**
     * Get Progress
     * @returns InferenceProgressSchemaOut Successful Response
     * @throws ApiError
     */
    public static getProgressGetInferenceProgressGet(): CancelablePromise<InferenceProgressSchemaOut> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/get_inference_progress',
        });
    }
    /**
     * Delete Image
     * @param imageId
     * @returns any Successful Response
     * @throws ApiError
     */
    public static deleteImageDeleteImgImageIdDelete(
        imageId: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/delete_img/{image_id}',
            path: {
                'image_id': imageId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Delete All Images
     * @returns any Successful Response
     * @throws ApiError
     */
    public static deleteAllImagesDeleteAllImagesDelete(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/delete_all_images',
        });
    }
    /**
     * Create Product
     * @param requestBody
     * @returns ProductSchemaOut Successful Response
     * @throws ApiError
     */
    public static createProductV1ProductsPost(
        requestBody: ProductSchemaIn,
    ): CancelablePromise<ProductSchemaOut> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/v1/products',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Product
     * @param productId
     * @returns ProductSchemaOut Successful Response
     * @throws ApiError
     */
    public static createProductV1ProductsProductIdGet(
        productId: string,
    ): CancelablePromise<ProductSchemaOut> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/v1/products/{product_id}',
            path: {
                'product_id': productId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
